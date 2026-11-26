from pydantic import Field

from .agents.forgen import evaluate_descriptor_rules
from .agents.mecins import feedback_requires_reassessment, surfactant_descriptor_criteria
from .schemas import Record


class CaseObservation(Record):
    surfactant: str
    inhibition_efficiency_pct: float = Field(le=100)


class PaperReflectionCase(Record):
    comparison: str
    sources: dict[str, str]
    reference: CaseObservation
    initial_feedback: CaseObservation
    follow_up: CaseObservation
    descriptors: dict[str, dict[str, float]]


def _screen(reference, candidates, criteria):
    results = []
    for name, values in candidates.items():
        score, checks = evaluate_descriptor_rules(reference, values, criteria)
        results.append({"candidate": name, "score": score, "checks": checks,
                        "passes": all(check["passes"] for check in checks)})
    results.sort(key=lambda item: (-item["score"], item["candidate"]))
    selected = next((item["candidate"] for item in results if item["passes"]), None)
    return {"criteria": [rule.model_dump() for rule in criteria], "comparisons": results, "selected": selected}


def replay_paper_reflection(case, catalog):
    reference = case.reference.surfactant
    candidates = {name: values for name, values in case.descriptors.items() if name != reference}
    groups = {row["name"]: int(row["group"]) for row in catalog if row["name"] in case.descriptors}
    if groups.keys() != case.descriptors.keys() or set(groups.values()) != {3}:
        raise ValueError("The case requires catalogued group-3 surfactants.")
    reference_values = case.descriptors[reference]
    initial = _screen(reference_values, candidates, surfactant_descriptor_criteria(False))
    if initial["selected"] != case.initial_feedback.surfactant:
        raise ValueError("The recorded feedback does not match the candidate selected by the initial screen.")

    baseline_ie = case.reference.inhibition_efficiency_pct
    trial_ie = case.initial_feedback.inhibition_efficiency_pct
    trigger = feedback_requires_reassessment(baseline_ie, [trial_ie])
    result = {
        "mode": "retrospective_rule_replay",
        "comparison": case.comparison,
        "sources": case.sources,
        "group": 3,
        "reference": case.reference.model_dump(),
        "initial_screen": initial,
        "feedback": {**case.initial_feedback.model_dump(),
                     "delta_vs_reference_pp": round(trial_ie - baseline_ie, 10),
                     "reassessment_triggered": trigger},
        "revised_screen": None,
        "reported_follow_up": None,
    }
    if trigger:
        revised = _screen(reference_values, candidates, surfactant_descriptor_criteria(True))
        result["revised_screen"] = revised
        follow_up = case.follow_up
        if revised["selected"] == follow_up.surfactant:
            result["reported_follow_up"] = {
                **follow_up.model_dump(),
                "delta_vs_reference_pp": round(follow_up.inhibition_efficiency_pct - baseline_ie, 10),
            }
    return result
