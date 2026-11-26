import asyncio
from datetime import datetime, timezone

from .agents import ExpOptAgent, ForGenAgent, MecInsAgent
from .agents.expopt import composition, family
from .agents.mecins import feedback_requires_reassessment
from .explain import analyze_shap
from .llm import ModelClient
from .rag import RagStore
from .schemas import MechanismHypothesis, WorkflowRecord, WorkflowState
from .storage import WorkflowStore


def make_orchestrator(settings):
    rag = RagStore(settings.rag_db)
    llm = ModelClient(settings)
    return CorALOrchestrator(
        ForGenAgent(rag, llm, settings.coral_cluster_priors, settings.coral_molecular_catalog),
        ExpOptAgent(settings.coral_bo_workers), MecInsAgent(llm, rag), WorkflowStore(settings.workflows_dir),
    )


class CorALOrchestrator:
    def __init__(self, forgen, expopt, mecins, store):
        self.forgen = forgen
        self.expopt = expopt
        self.mecins = mecins
        self.store = store
        self.locks = [asyncio.Lock() for _ in range(32)]

    @staticmethod
    def _event(record, agent, action, **payload):
        record.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent, "action": action, "payload": payload,
        })

    @staticmethod
    def _store_sources(record, sources):
        known = {item.source_id for item in record.sources}
        record.sources.extend(item for item in sources if item.source_id not in known)

    @staticmethod
    def _model_call(record, agent, operation, call):
        if call:
            record.model_calls.append({
                **call, "agent": agent, "operation": operation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def _queue(self, record, formulations, phase):
        protocols = self.expopt.create_protocols(formulations, record.request.context)
        record.formulations.extend(formulations)
        record.protocols.extend(protocols)
        record.pending_formulation_ids = [item.formulation_id for item in formulations]
        record.phase = phase
        record.state = (WorkflowState.REDESIGN_PROPOSED if phase == "redesign" else WorkflowState.AWAITING_EXPERIMENT)
        self._event(record, "ExpOpt", "protocols_generated", phase=phase,
                    protocol_ids=[item.protocol_id for item in protocols])

    async def start(self, request):
        record = WorkflowRecord(request=request)
        self._event(record, "Orchestrator", "workflow_created")
        formulations, sources, call = await self.forgen.recommend(
            request.question, request.context, request.number_of_candidates,
        )
        self._model_call(record, "ForGen", "formulation_generation", call)
        self._store_sources(record, sources)
        self._event(record, "ForGen", "candidates_generated",
                    model_mode="online" if self.forgen.llm.enabled else "offline",
                    formulation_ids=[item.formulation_id for item in formulations],
                    retrieved_source_ids=[item.source_id for item in sources])
        self._queue(record, formulations, "screen")
        return self.store.save(record)

    async def submit_results(self, workflow_id, request):
        async with self.locks[hash(workflow_id.lower()) % len(self.locks)]:
            record = self.store.load(workflow_id)
            submitted = {item.formulation_id: item for item in request.results}
            if len(submitted) != len(request.results):
                raise ValueError("Submit one result per formulation.")
            measured = {item.formulation_id: item for item in record.experiment_results}
            if all(measured.get(key) == result for key, result in submitted.items()):
                return record
            if record.state not in {WorkflowState.AWAITING_EXPERIMENT, WorkflowState.REDESIGN_PROPOSED}:
                raise ValueError(f"Workflow state {record.state} cannot accept results.")
            all_formulations = {item.formulation_id: item for item in record.formulations + record.redesign_formulations}
            pending = set(record.pending_formulation_ids)
            if not pending:
                pending = set(all_formulations) - measured.keys()
            if submitted.keys() - pending:
                raise ValueError("Measurements must match the active batch.")
            protocol_formulations = {item.protocol_id: item.formulation.formulation_id for item in record.protocols}
            if any(item.laboratory_task_id and not item.protocol_id for item in submitted.values()):
                raise ValueError("A laboratory task result must identify the corresponding protocol.")
            if any(item.protocol_id and protocol_formulations.get(item.protocol_id) != item.formulation_id
                   for item in submitted.values()):
                raise ValueError("The submitted protocol does not match the formulation.")
            if any(key in measured and measured[key] != result for key, result in submitted.items()):
                raise ValueError("A stored measurement cannot be overwritten.")
            new_results = [item for key, item in submitted.items() if key not in measured]
            record.experiment_results.extend(new_results)
            measured.update(submitted)
            if new_results:
                self._event(record, "ExpOpt", "measurements_received", measurements=[
                    {"formulation_id": item.formulation_id, "protocol_id": item.protocol_id,
                     "laboratory_task_id": item.laboratory_task_id} for item in new_results
                ])
            if pending - measured.keys():
                return self.store.save(record)

            record.pending_formulation_ids = []
            action = request.next_step or ("redesign" if request.request_redesign else "complete")
            best = max((all_formulations[key] for key in measured),
                       key=lambda item: measured[item.formulation_id].inhibition_efficiency)
            if request.reference_formulation_id and request.reference_formulation_id not in measured:
                raise ValueError("The reference must be a measured formulation from this workflow.")
            requested = all_formulations.get(request.reference_formulation_id)
            if action == "seed_ratios":
                if record.phase not in {"screen", "redesign"}:
                    raise ValueError("Seed ratios can follow formulation screening or substitution trials.")
                reference = requested or max((all_formulations[key] for key in sorted(pending)),
                                             key=lambda item: measured[item.formulation_id].inhibition_efficiency)
                record.ratio_reference_id = reference.formulation_id
                observed = [composition(all_formulations[key], reference) for key in measured
                            if family(all_formulations[key]) == family(reference)]
                feedback = record.mechanism_rounds[-1] if record.phase == "redesign" else None
                proposals, call = await self.forgen.propose_seed_ratios(reference, observed, feedback)
                self._model_call(record, "ForGen", "ratio_seed_generation", call)
                self._event(record, "ForGen", "seed_ratios_generated",
                            reference_id=reference.formulation_id, count=len(proposals))
                self._queue(record, proposals, "ratio_seed")
            elif action == "optimize":
                reference = requested or all_formulations.get(record.ratio_reference_id, best)
                proposals, analysis = await asyncio.to_thread(
                    self.expopt.propose_ratios, reference, list(all_formulations.values()), record.experiment_results,
                )
                record.ratio_reference_id = reference.formulation_id
                record.optimization_rounds.append(analysis)
                self._event(record, "ExpOpt", "expected_improvement_ranked", round=len(record.optimization_rounds))
                self._queue(record, proposals, "bayesian_optimization")
            elif action == "redesign":
                await self._redesign(record, all_formulations, measured, pending, requested or best,
                                     request.target_molecule_id, request.substitution_pool_ids)
            else:
                record.state = WorkflowState.COMPLETE
                self._event(record, "Orchestrator", "workflow_completed", best_formulation_id=best.formulation_id)
            return self.store.save(record)

    async def _redesign(self, record, formulations, measured, active, base, target_id, shortlist):
        previous = record.mechanism_rounds[-1] if record.mechanism_rounds else None
        trials, prior, excluded = [], None, set()
        if record.phase == "redesign" and previous:
            prior = MechanismHypothesis.model_validate(previous["hypothesis"])
            reference = formulations[prior.reference_formulation_id]
            trials = [measured[key] for key in sorted(active)]
            if feedback_requires_reassessment(measured[reference.formulation_id].inhibition_efficiency,
                                              (item.inhibition_efficiency for item in trials)):
                base = reference
                excluded = {item for step in record.mechanism_rounds
                            if step["hypothesis"]["reference_formulation_id"] == base.formulation_id
                            for item in step["screening"]["selected_molecule_ids"]}
            else:
                trials, prior = [], None
        x, y = self.expopt.observations_for(base, list(formulations.values()), record.experiment_results)
        shap_summary = None
        if len(x) >= 10:
            feature_order = [item.name for item in base.components]
            report = await asyncio.to_thread(analyze_shap, x, y, feature_order)
            shap_summary = {key: report[key] for key in (
                "feature_order", "test_rmse", "mean_absolute_shap", "low_and_high_fraction_mean_shap",
            )}
        decision, sources, analysis, call = await self.mecins.analyze(
            base, measured[base.formulation_id], trials, self.forgen.catalog, prior, shap_summary, target_id,
        )
        self._model_call(record, "MecIns", "descriptor_reassessment" if analysis["no_improvement"] else "hypothesis", call)
        self._store_sources(record, sources)
        if shortlist is None and prior:
            shortlist = previous["screening"].get("substitution_pool_ids")
        proposals, screening = await asyncio.to_thread(self.forgen.redesign, base, decision, excluded, shortlist)
        record.mechanism_hypotheses.append(decision)
        record.mechanism_rounds.append({
            "hypothesis": decision.model_dump(mode="json"), "analysis": analysis,
            "screening": screening, "proposal_ids": [item.formulation_id for item in proposals],
        })
        self._event(record, "MecIns", "hypothesis_reassessed" if analysis["no_improvement"] else "hypothesis_generated",
                    reference_id=base.formulation_id, retrieved_source_ids=[item.source_id for item in sources])
        self._event(record, "ForGen", "same_group_screening", candidate_count=len(proposals))
        if proposals:
            record.redesign_formulations = proposals
            self._queue(record, proposals, "redesign")
        else:
            record.state = WorkflowState.MECINS_COMPLETE
