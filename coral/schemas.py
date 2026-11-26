from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class WorkflowState(StrEnum):
    CREATED = "created"
    FORGEN_COMPLETE = "forgen_complete"
    AWAITING_EXPERIMENT = "awaiting_experiment"
    MECINS_COMPLETE = "mecins_complete"
    REDESIGN_PROPOSED = "redesign_proposed"
    COMPLETE = "complete"
    FAILED = "failed"


class ResearchContext(Record):
    substrate: str = "Q235 carbon steel"
    medium: str = "3.5 wt% NaCl"
    temperature_c: float = 25.0
    ph: float | None = None
    target_concentration: str = "user-defined"
    constraints: list[str] = Field(default_factory=list)


class Component(Record):
    name: str
    cluster: int = Field(ge=1, le=11)
    fraction: float = Field(ge=0, le=1)
    molecule_id: str | None = None


class Formulation(Record):
    formulation_id: str = Field(default_factory=lambda: f"F-{uuid4().hex[:8]}")
    components: list[Component] = Field(min_length=2)
    rationale: str
    source_ids: list[str] = Field(default_factory=list)
    status: str = "candidate"

    @model_validator(mode="after")
    def fractions_sum_to_one(self):
        total = sum(component.fraction for component in self.components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Component fractions must sum to 1.0, got {total}")
        names = [component.name.strip().casefold() for component in self.components]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("Component identities must be nonempty and distinct.")
        return self


class SourceEvidence(Record):
    source_id: str
    document_path: str
    chunk_number: int
    content: str
    score: float


class ExperimentProtocol(Record):
    protocol_id: str = Field(default_factory=lambda: f"P-{uuid4().hex[:8]}")
    formulation: Formulation
    context: ResearchContext
    comparison_groups: list[str] = Field(
        default_factory=lambda: ["blank", "single_components", "cocktail"]
    )
    measurements: list[str] = Field(
        default_factory=lambda: ["OCP", "EIS", "PDP"]
    )
    working_electrode_area_cm2: float = 0.785
    counter_electrode: str = "Platinum, 1.0 cm2"
    reference_electrode: str = "Saturated calomel electrode"
    immersion_hours: float = 24
    eis_frequency_hz: tuple[float, float] = (100000, 0.01)
    eis_amplitude_mv: float = 5
    pdp_range_v_relative_to_ocp: tuple[float, float] = (-0.3, 0.3)
    pdp_scan_rate_mv_per_s: float = 0.167


class ExperimentResult(Record):
    formulation_id: str
    protocol_id: str | None = Field(default=None, min_length=1)
    laboratory_task_id: str | None = Field(default=None, min_length=1)
    inhibition_efficiency: float = Field(le=100)
    synergy_index: float | None = None
    eis_rct_ohm_cm2: float | None = Field(default=None, ge=0)
    chi_square: float | None = Field(default=None, ge=0)
    observations: dict[str, Any] = Field(default_factory=dict)
    source_reference: str = "Submitted experiment"


class DescriptorCriterion(Record):
    descriptor: str
    direction: Literal["lower", "higher", "near"]
    tolerance: float | None = Field(default=None, gt=0)


class MechanismHypothesis(Record):
    hypothesis: str
    supporting_observations: list[str]
    uncertainty: str
    validation_actions: list[str]
    redesign_strategy: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    reference_formulation_id: str | None = None
    target_molecule_id: str | None = None
    descriptor_criteria: list[DescriptorCriterion] = Field(default_factory=list)


class StartWorkflowRequest(Record):
    question: str = Field(min_length=1)
    context: ResearchContext = Field(default_factory=ResearchContext)
    number_of_candidates: int = Field(default=10, ge=1, le=30)


class SubmitResultsRequest(Record):
    results: list[ExperimentResult] = Field(min_length=1)
    request_redesign: bool = True
    next_step: Literal["seed_ratios", "optimize", "redesign", "complete"] | None = None
    reference_formulation_id: str | None = None
    target_molecule_id: str | None = None
    substitution_pool_ids: list[str] | None = Field(default=None, min_length=1)


class WorkflowRecord(Record):
    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: WorkflowState = WorkflowState.CREATED
    request: StartWorkflowRequest
    sources: list[SourceEvidence] = Field(default_factory=list)
    formulations: list[Formulation] = Field(default_factory=list)
    protocols: list[ExperimentProtocol] = Field(default_factory=list)
    experiment_results: list[ExperimentResult] = Field(default_factory=list)
    mechanism_hypotheses: list[MechanismHypothesis] = Field(default_factory=list)
    redesign_formulations: list[Formulation] = Field(default_factory=list)
    pending_formulation_ids: list[str] = Field(default_factory=list)
    phase: str = "screen"
    ratio_reference_id: str | None = None
    optimization_rounds: list[dict[str, Any]] = Field(default_factory=list)
    mechanism_rounds: list[dict[str, Any]] = Field(default_factory=list)
    model_calls: list[dict[str, Any]] = Field(default_factory=list)
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

