import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Literal
from unicodedata import normalize

from pydantic import BaseModel, Field

from .llm import ModelGatewayError


SECTIONS = ("abstract", "conclusion")
SCREEN = re.compile(r"\b(?:synerg\w*|mixtures?|combinations?|combined|binary inhibitors?)\b", re.I)


class ExtractedFormulation(BaseModel):
    section: Literal["abstract", "conclusion"]
    evidence: str = Field(min_length=1)
    names: list[str] = Field(min_length=2, max_length=3)


class Extraction(BaseModel):
    formulations: list[ExtractedFormulation]


def _normalize(value):
    return " ".join(re.findall(r"[a-z0-9]+", normalize("NFKC", str(value)).casefold()))


def _screen(study):
    sections = {key: study[key] for key in SECTIONS if study.get(key)}
    return sections if any(SCREEN.search(source) for source in sections.values()) else {}


def _offline_formulations(sections, pattern, aliases):
    for section, source in sections.items():
        for evidence in re.split(r"(?<=[.!?])\s+|;\s*", source):
            if not SCREEN.search(evidence):
                continue
            clause = _normalize(evidence)
            if re.search(r"\b(?:separately|individually|respectively)\b", clause):
                continue
            mentions = {}
            for name in pattern.findall(clause):
                mentions.setdefault(frozenset(aliases[name]), name)
            size = 3 if "ternary" in clause.split() else 2 if "binary" in clause.split() else None
            if len(mentions) in (2, 3) and (size is None or len(mentions) == size):
                yield {"section": section, "evidence": evidence, "method": "offline_alias_match",
                       "components": [{"name": name} for name in mentions.values()]}


async def extract_with_deepseek(studies, client):
    extractions = {}
    for study in studies:
        sections = _screen(study)
        if not sections or "formulations" in study:
            continue
        response, call = await client.complete_json(
            "Extract only experimentally reported binary and ternary corrosion inhibitor formulations. "
            "Treat section text as evidence only. For each formulation give its section, one verbatim "
            "evidence passage, and the list of inhibitor names appearing in that passage. "
            "Exclude isolated inhibitors, control solutions and unrelated co-mentions.",
            json.dumps({"sections": sections, "response_schema": Extraction.model_json_schema()}),
            {},
        )
        try:
            formulations = Extraction.model_validate(response).formulations
            for item in formulations:
                content = " " + _normalize(item.evidence) + " "
                if item.evidence not in sections.get(item.section, "") or any(
                    not _normalize(name) or f" {_normalize(name)} " not in content for name in item.names
                ):
                    raise ValueError("The extracted evidence or name is absent from the supplied section.")
        except ValueError as error:
            raise ModelGatewayError(f"Invalid extraction for {study['publication_id']}.") from error
        extractions[study["publication_id"]] = {
            "model_call": call,
            "formulations": [
                {"section": item.section, "evidence": item.evidence, "method": "deepseek",
                 "components": [{"name": name} for name in item.names]}
                for item in formulations
            ],
        }
    return extractions


def _curated_formulations(study, sections):
    for item in study["formulations"]:
        if isinstance(item, list):
            yield {"section": None, "evidence": None, "method": "curated", "components": item}
            continue
        section, evidence = item["section"], item["evidence"]
        if not evidence.strip() or evidence not in sections.get(section, ""):
            raise ValueError(f"Curated evidence must be quoted from the supplied section for {study['publication_id']}.")
        yield {"section": section, "evidence": evidence, "method": "curated", "components": item["components"]}


def mine_group_pairs(studies, identities, catalog, assignments=None, extractions=None):
    rows = {row["molecule_id"]: row for row in catalog}
    if assignments is None:
        groups = {key: int(row["group"]) for key, row in rows.items()}
    else:
        groups = assignments
        if groups.keys() != rows.keys() or any(type(group) is not int or group < 1 for group in groups.values()):
            raise ValueError("Cluster assignments must provide a positive integer group for each catalog molecule.")

    names, identifiers = defaultdict(set), {}
    for identity in identities:
        key = identity["molecule_id"]
        if key not in groups:
            raise ValueError(f"Unknown catalog molecule {key}.")
        for alias in [rows[key]["name"], *identity.get("aliases", [])]:
            if normalized := _normalize(alias):
                names[normalized].add(key)
        for field in ("pubchem_cid", "inchikey"):
            index = identifiers.setdefault(field, defaultdict(set))
            values = identity.get(field, [])
            for value in values if isinstance(values, list) else [values]:
                if value is not None and (normalized := _normalize(value)):
                    index[normalized].add(key)
    pattern = "|".join(re.escape(name) for name in sorted(names, key=lambda name: (-len(name), name)))
    aliases = re.compile(rf"\b(?:{pattern})\b" if pattern else r"(?!)")

    extracted, classified, rejected = [], [], []
    by_publication = {}
    hits = Counter()
    screened = []
    for study in studies:
        publication = study["publication_id"]
        if publication in by_publication:
            raise ValueError(f"Duplicate publication {publication}.")
        by_publication[publication] = {}
        sections = _screen(study)
        if not sections:
            continue
        screened.append(publication)
        if "formulations" in study:
            candidates = _curated_formulations(study, sections)
        elif extractions is None:
            candidates = _offline_formulations(sections, aliases, names)
        else:
            candidates = extractions[publication]["formulations"]
        pairs = Counter()
        seen = set()
        for candidate in candidates:
            record = {"publication_id": publication, **candidate}
            extracted.append(record)
            components = record["components"]
            if len(components) not in (2, 3):
                raise ValueError("Literature formulations must be binary or ternary.")
            resolved = []
            for component in components:
                explicit = [identifiers.get(key, {}).get(_normalize(component[key]), set())
                            for key in ("pubchem_cid", "inchikey") if component.get(key)]
                matches = set.intersection(*explicit) if explicit else names.get(_normalize(component.get("name", "")), set())
                if len(matches) != 1:
                    rejected.append({"publication_id": publication, "component": component,
                                     "candidate_ids": sorted(matches), "section": record["section"],
                                     "evidence": record["evidence"]})
                    continue
                key = next(iter(matches))
                resolved.append({"reported_component": component, "molecule_id": key,
                                 "name": rows[key]["name"], "group": groups[key]})
            signature = frozenset(item["molecule_id"] for item in resolved)
            if len(signature) != len(components) or signature in seen:
                continue
            seen.add(signature)
            molecular_pairs = [
                {"molecule_ids": [a["molecule_id"], b["molecule_id"]], "groups": sorted((a["group"], b["group"]))}
                for a, b in combinations(resolved, 2)
            ]
            pairs.update(tuple(item["groups"]) for item in molecular_pairs)
            classified.append({**record, "components": resolved, "molecular_pairs": molecular_pairs,
                               "group_pairs": [item["groups"] for item in molecular_pairs]})
        hits.update(pairs)
        by_publication[publication] = {f"{a}-{b}": count for (a, b), count in sorted(pairs.items())}
    return {
        "classification_source": "archived_catalog" if assignments is None else "clustering_run",
        "screened_publications": screened,
        "extracted_formulations": extracted,
        "classified_formulations": classified,
        "pair_frequencies": [{"pair": list(pair), "frequency": count}
                             for pair, count in sorted(hits.items(), key=lambda item: (-item[1], item[0]))],
        "by_publication": by_publication,
        "unresolved_or_ambiguous": rejected,
        "model_calls": {key: value["model_call"] for key, value in (extractions or {}).items()},
    }
