import numpy as np

from ..optimization import rank_compositions
from ..schemas import ExperimentProtocol, Formulation


def family(formulation):
    return frozenset(component.molecule_id or component.name.casefold() for component in formulation.components)


def composition(formulation, reference):
    fractions = {item.molecule_id or item.name.casefold(): item.fraction for item in formulation.components}
    return [round(100 * fractions[item.molecule_id or item.name.casefold()], 8)
            for item in reference.components]


def with_ratio(reference, row, rationale, status):
    return Formulation(
        components=[component.model_copy(update={"fraction": float(percent) / 100})
                    for component, percent in zip(reference.components, row, strict=True)],
        rationale=rationale, status=status, source_ids=reference.source_ids,
    )


class ExpOptAgent:
    def __init__(self, workers=1):
        self.workers = workers

    def create_protocols(self, formulations, context):
        return [ExperimentProtocol(formulation=item, context=context) for item in formulations]

    def propose_ratios(self, reference, formulations, results):
        eligible = {item.formulation_id: item for item in formulations if family(item) == family(reference)}
        measured = [result for result in results if result.formulation_id in eligible]
        seed_and_bo = [result for result in measured
                       if eligible[result.formulation_id].status in {"ratio_seed", "optimization_candidate"}]
        training = seed_and_bo if any(eligible[item.formulation_id].status == "ratio_seed"
                                      for item in seed_and_bo) else measured
        analysis = rank_compositions(
            [composition(eligible[result.formulation_id], reference) for result in training],
            [result.inhibition_efficiency for result in training],
            excluded=[composition(eligible[result.formulation_id], reference) for result in measured],
            workers=self.workers,
        )
        analysis["feature_order"] = [item.name for item in reference.components]
        analysis["training_formulation_ids"] = [result.formulation_id for result in training]
        proposals = [
            with_ratio(reference, row["composition_pct"], "Selected by expected improvement.", "optimization_candidate")
            for row in analysis["proposals"]
        ]
        for row, proposal in zip(analysis["proposals"], proposals, strict=True):
            row["formulation_id"] = proposal.formulation_id
        return proposals, analysis

    @staticmethod
    def observations_for(reference, formulations, results):
        same_family = {item.formulation_id: item for item in formulations if family(item) == family(reference)}
        eligible = [item for item in results if item.formulation_id in same_family]
        x = np.array([composition(same_family[item.formulation_id], reference) for item in eligible])
        return x, [item.inhibition_efficiency for item in eligible]
