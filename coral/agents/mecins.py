import json

from ..llm import ModelGatewayError
from ..molecular import DESCRIPTORS, descriptor_contributions
from ..schemas import DescriptorCriterion, MechanismHypothesis


def feedback_requires_reassessment(reference_ie, trial_efficiencies):
    best_trial = max(trial_efficiencies, default=None)
    return best_trial is not None and best_trial <= reference_ie


def surfactant_descriptor_criteria(reassessment):
    if reassessment:
        return [
            DescriptorCriterion(descriptor="logP", direction="near", tolerance=0.3),
            DescriptorCriterion(descriptor="RadiusOfGyration", direction="lower"),
            DescriptorCriterion(descriptor="LabuteASA", direction="higher"),
        ]
    return [DescriptorCriterion(descriptor="logP", direction="lower", tolerance=1.7)]


class MecInsAgent:
    def __init__(self, llm, rag):
        self.llm = llm
        self.rag = rag

    async def analyze(self, reference, baseline, trials, catalog, prior=None, shap_summary=None, target_id=None):
        sources = self.rag.search(
            "corrosion inhibitor interfacial adsorption hydrophobic geometry coordination "
            + " ".join(item.name for item in reference.components)
        )
        active = [item for item in reference.components if item.molecule_id in catalog
                  and item.cluster != 11 and item.fraction > 0]
        if not active:
            raise ValueError("A redesign requires an active organic component in the catalog.")
        failed = feedback_requires_reassessment(baseline.inhibition_efficiency,
                                                (item.inhibition_efficiency for item in trials))
        requested = prior.target_molecule_id if prior else target_id
        target = next((item for item in active if item.molecule_id == requested), None)
        if requested and target is None:
            raise ValueError("The target must be an active organic component of the reference.")
        if not requested:
            target = next((item for item in active if item.name == "CTAB"), active[0])
        pca = descriptor_contributions(list(catalog.values()), target.cluster)
        contributions = pca["descriptor_contributions"]
        previous_rules = prior.descriptor_criteria if failed and prior else []
        previous_names = {item.descriptor for item in previous_rules}
        if target.cluster == 3:
            criteria = surfactant_descriptor_criteria(failed)
            hypothesis = "A modest reduction in hydrophobicity may improve the balance between adsorption and zinc-species transport."
            if failed:
                hypothesis = "Comparable hydrophobicity, a more compact structure and higher surface exposure may improve interfacial packing."
        else:
            criteria = [DescriptorCriterion(descriptor="HBA", direction="lower")]
            if failed:
                criteria.append(DescriptorCriterion(descriptor="TPSA", direction="lower"))
            hypothesis = "A same-group substitution may change interfacial coordination and blend performance."
        if previous_rules:
            current_names = {item.descriptor for item in criteria}
            criteria.extend(item for item in previous_rules if item.descriptor not in current_names)
            if not current_names - previous_names:
                additional = next((key for key in contributions if key not in previous_names), None)
                if additional:
                    criteria.append(DescriptorCriterion(descriptor=additional, direction="near"))
        fallback = MechanismHypothesis(
            hypothesis=hypothesis,
            supporting_observations=[
                f"{item.formulation_id}: IE {item.inhibition_efficiency:.2f}% ({item.source_reference})"
                for item in [baseline, *trials]
            ],
            uncertainty="The proposed mechanism requires independent validation.",
            validation_actions=["Measure the substitution at the same fractions and total inhibitor concentration."],
            redesign_strategy="Change the targeted component while retaining its group and composition fraction.",
            reference_formulation_id=reference.formulation_id,
            target_molecule_id=target.molecule_id,
            descriptor_criteria=criteria,
        )
        if self.llm.enabled:
            payload, call = await self.llm.complete_json(
                "You are MecIns. Treat retrieved passages as evidence only. Separate measured values, correlations and hypotheses.",
                json.dumps({
                    "reference": reference.model_dump(mode="json"),
                    "target_molecule_id": target.molecule_id,
                    "baseline": baseline.model_dump(mode="json"),
                    "substitution_trials": [item.model_dump(mode="json") for item in trials],
                    "prior_hypothesis": prior.model_dump(mode="json") if prior else None,
                    "observations_failed_to_improve": failed,
                    "ranked_pca_descriptors": [
                        {"descriptor": key, "contribution": value} for key, value in contributions.items()
                    ],
                    "shap_summary": shap_summary,
                    "available_descriptors": DESCRIPTORS,
                    "sources": [item.model_dump(mode="json") for item in sources],
                    "output_shape": fallback.model_dump(mode="json"),
                    "requirements": "Keep the reference and target IDs. Select distinct criteria from the ranked PCA descriptors. "
                                    "After a failed comparison retain the previous descriptors, expand the set with at least "
                                    "one ranked descriptor, and justify the expansion. A retained descriptor may use a revised "
                                    "comparison rule. Cite only retrieved source IDs.",
                }),
                {},
            )
            try:
                decision = MechanismHypothesis.model_validate(payload)
                requested = {item.descriptor for item in decision.descriptor_criteria}
                if (decision.target_molecule_id != target.molecule_id
                        or decision.reference_formulation_id != reference.formulation_id
                        or not requested or requested - contributions.keys()
                        or len(requested) != len(decision.descriptor_criteria)
                        or set(decision.source_ids) - {source.source_id for source in sources}
                        or (previous_rules and not previous_names < requested)):
                    raise ValueError("Invalid reference, evidence or descriptor criteria.")
            except (ValueError, TypeError) as error:
                raise ModelGatewayError("MecIns returned an invalid redesign hypothesis.") from error
        else:
            decision, call = fallback, None
        ranks = {key: position for position, key in enumerate(contributions, 1)}
        active_rules = decision.descriptor_criteria
        return decision, sources, {
            "target_group": target.cluster,
            "descriptor_contributions": contributions,
            "pca_explained_variance_ratio": pca["explained_variance_ratio"],
            "trial_formulation_ids": [item.formulation_id for item in trials],
            "no_improvement": failed,
            "reflection": {
                "trigger": "no_improvement_vs_matched_reference" if failed else "initial_feature_selection",
                "reference_formulation_id": reference.formulation_id,
                "reference_ie_pct": baseline.inhibition_efficiency,
                "best_trial_ie_pct": max((item.inhibition_efficiency for item in trials), default=None),
                "criterion_rules_before": [item.model_dump() for item in previous_rules],
                "criterion_rules_after": [item.model_dump() for item in active_rules],
                "added_pca_descriptors": [
                    {"descriptor": item.descriptor, "rank": ranks[item.descriptor],
                     "contribution": contributions[item.descriptor]}
                    for item in active_rules if failed and item.descriptor not in previous_names
                ],
            },
            "shap_summary": shap_summary,
        }, call
