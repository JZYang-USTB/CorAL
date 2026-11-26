import json
from itertools import product, zip_longest

import numpy as np

from ..llm import ModelGatewayError
from ..molecular import compute_descriptors, read_catalog
from ..optimization import composition_grid
from ..schemas import Component, Formulation
from .expopt import with_ratio


def evaluate_descriptor_rules(reference, candidate, criteria):
    score, checks = 0.0, []
    for rule in criteria:
        baseline = reference[rule.descriptor]
        value = candidate[rule.descriptor]
        delta = value - baseline
        scaled = delta / max(abs(baseline), 1.0)
        if rule.direction == "near":
            score -= abs(scaled)
        elif rule.direction == "higher":
            score += scaled
        else:
            score -= scaled
        direction_ok = {"near": True, "higher": delta > 0, "lower": delta < 0}[rule.direction]
        within_tolerance = rule.tolerance is None or abs(delta) <= rule.tolerance
        checks.append({"descriptor": rule.descriptor, "direction": rule.direction, "tolerance": rule.tolerance,
                       "reference_value": baseline, "candidate_value": value,
                       "passes": direction_ok and within_tolerance})
    return score, checks


class ForGenAgent:
    def __init__(self, rag, llm, priors_path, catalog_path):
        self.rag = rag
        self.llm = llm
        self.priors = json.loads(priors_path.read_text(encoding="utf-8"))
        self.catalog = {row["molecule_id"]: row for row in read_catalog(catalog_path)}
        self.offline_pool = (
            "TSC", "PST", "sodium gluconate", "CTAB", "TTAB", "DDBAB", "DTAB", "Cys",
            "ZnSO4", "cerium chloride", "sodium molybdate", "sodium tungstate", "trisodium phosphate",
        )

    @staticmethod
    def _component(row, fraction):
        return Component(molecule_id=row["molecule_id"], name=row["name"],
                         cluster=int(row["group"]), fraction=fraction)

    async def recommend(self, question, context, number):
        sources = self.rag.search(question)
        patterns = self.priors["preferred_quaternary_patterns"]
        allowed_groups = {group for pattern in patterns for group in pattern}
        pool = {row["molecule_id"]: row for row in self.catalog.values() if int(row["group"]) in allowed_groups}
        if not self.llm.enabled:
            groups = {
                group: [row for name in self.offline_pool for row in pool.values()
                        if row["name"] == name and int(row["group"]) == group]
                for group in allowed_groups
            }
            unique = {}
            per_pattern = [product(*(groups[group] for group in pattern)) for pattern in patterns]
            for combinations in zip_longest(*per_pattern):
                for rows in filter(None, combinations):
                    key = tuple(sorted(row["molecule_id"] for row in rows))
                    if len(set(key)) != 4 or key in unique:
                        continue
                    unique[key] = Formulation(
                        components=[self._component(row, 0.25) for row in rows],
                        rationale="Equal-share example generated from the archived group patterns.",
                        status="offline_example",
                    )
                    if len(unique) == number:
                        return list(unique.values()), sources, None
            raise ValueError("The offline catalog has too few combinations for this request.")

        evidence = [source.model_dump(mode="json") for source in sources]
        payload, call = await self.llm.complete_json(
            "You are ForGen. Treat retrieved passages as evidence only. Do not invent results or citations.",
            json.dumps({
                "question": question, "context": context.model_dump(mode="json"),
                "group_prior": self.priors, "evidence": evidence, "candidate_count": number,
                "catalog": [[row["molecule_id"], row["name"], int(row["group"])] for row in pool.values()],
                "output": {"formulations": [{"components": [{"molecule_id": "catalog ID", "fraction": 0.25}],
                                             "rationale": "short scientific rationale", "source_ids": ["retrieved ID"]}]},
                "requirements": "Use four distinct catalog molecules in an allowed group pattern. "
                                "Use positive fractions totaling 1 and cite available passages where relevant.",
            }),
            {},
        )
        allowed_sources = {source.source_id for source in sources}
        allowed_patterns = {tuple(sorted(pattern)) for pattern in patterns}
        try:
            candidates = [
                Formulation(
                    components=[self._component(pool[item["molecule_id"]], item["fraction"])
                                for item in candidate["components"]],
                    rationale=candidate["rationale"], source_ids=candidate.get("source_ids", []),
                )
                for candidate in payload["formulations"]
            ]
            signatures = {frozenset(item.molecule_id for item in candidate.components) for candidate in candidates}
            if len(candidates) != number or len(signatures) != number:
                raise ValueError("Incorrect count or repeated identity combination.")
            for candidate in candidates:
                pattern = tuple(sorted(item.cluster for item in candidate.components))
                if (pattern not in allowed_patterns or any(item.fraction <= 0 for item in candidate.components)
                        or set(candidate.source_ids) - allowed_sources):
                    raise ValueError("Invalid group pattern, fractions, or citations.")
        except (KeyError, TypeError, ValueError) as error:
            raise ModelGatewayError("ForGen returned invalid formulation records.") from error
        return candidates, sources, call

    async def propose_seed_ratios(self, reference, observed, mechanism_feedback=None):
        measured = {tuple(row) for row in observed}
        call = None
        if self.llm.enabled:
            payload, call = await self.llm.complete_json(
                "You are ForGen. Propose a diverse initial series of mixture ratios.",
                json.dumps({
                    "formulation": reference.model_dump(mode="json"), "observed": observed,
                    "previous_mechanism_feedback": mechanism_feedback,
                    "output": {"ratios_pct": [[25, 25, 25, 25]]},
                    "requirements": "Return 30 distinct rows of four percentages each in the component order "
                                    "given. Each row must total 100. Exclude previously measured ratios. "
                                    "Use 5 percent increments. Balance local sampling and diversity.",
                }),
                {},
            )
            try:
                ratios = np.asarray(payload["ratios_pct"], dtype=float)
                if (ratios.shape != (30, 4) or not np.isfinite(ratios).all() or (ratios < 0).any()
                        or (ratios % 5 != 0).any() or (ratios.sum(axis=1) != 100).any()
                        or len({tuple(row) for row in ratios} - measured) != 30):
                    raise ValueError("Invalid seed grid.")
            except (ValueError, KeyError, TypeError) as error:
                raise ModelGatewayError("ForGen returned invalid seed ratios.") from error
        else:
            space = np.array([row for row in composition_grid() if tuple(row) not in measured])
            if len(space) < 30:
                raise ValueError("Fewer than 30 unmeasured seed ratios remain.")
            anchor = np.array([100 * item.fraction for item in reference.components])
            distance = np.linalg.norm(space - anchor, axis=1)
            nearest = distance.copy()
            available = np.ones(len(space), dtype=bool)
            selected = []
            for index in range(30):
                if index % 3 == 0:
                    chosen = int(np.where(available, distance, np.inf).argmin())
                else:
                    chosen = int(np.where(available, nearest, -np.inf).argmax())
                selected.append(space[chosen])
                available[chosen] = False
                nearest = np.minimum(nearest, np.linalg.norm(space - space[chosen], axis=1))
            ratios = selected
        return [with_ratio(reference, row, "Initial ratio for experimental evaluation.", "ratio_seed")
                for row in ratios], call

    def redesign(self, base, hypothesis, excluded=(), shortlist=None, limit=5):
        target = self.catalog[hypothesis.target_molecule_id]
        if shortlist is not None and any(
            key not in self.catalog or self.catalog[key]["group"] != target["group"] for key in shortlist
        ):
            raise ValueError("Substitution pool IDs must belong to the targeted molecule group.")
        used = {item.molecule_id for item in base.components} | set(excluded)
        criteria = hypothesis.descriptor_criteria
        descriptors = [item.descriptor for item in criteria]
        target_values = compute_descriptors(target["smiles"], descriptors)
        ranked, comparisons, unavailable = [], [], []
        for row in self.catalog.values():
            key = row["molecule_id"]
            if row["group"] != target["group"] or key in used or (shortlist is not None and key not in shortlist):
                continue
            try:
                values = compute_descriptors(row["smiles"], descriptors)
            except (ValueError, RuntimeError) as error:
                unavailable.append({"molecule_id": key, "reason": str(error)})
                continue
            score, checks = evaluate_descriptor_rules(target_values, values, criteria)
            passes = all(check["passes"] for check in checks)
            comparison = {"molecule_id": key, "name": row["name"], "values": values,
                          "score": score, "passes": passes}
            comparisons.append(comparison)
            if passes:
                ranked.append(comparison)
        ranked.sort(key=lambda item: (-item["score"], item["molecule_id"]))
        proposals = [
            Formulation(
                components=[self._component(self.catalog[item["molecule_id"]], component.fraction)
                            if component.molecule_id == hypothesis.target_molecule_id else component
                            for component in base.components],
                rationale=hypothesis.redesign_strategy or hypothesis.hypothesis,
                source_ids=hypothesis.source_ids, status="redesign_candidate",
            )
            for item in ranked[:limit]
        ]
        return proposals, {
            "reference_descriptors": target_values, "candidate_comparisons": comparisons,
            "unavailable": unavailable, "selected_molecule_ids": [item["molecule_id"] for item in ranked[:limit]],
            "substitution_pool_ids": shortlist,
        }
