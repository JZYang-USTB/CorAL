import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from .clustering import cluster_catalog
from .config import DATA, Settings
from .eis import fit_eis, inhibition_efficiency_at_001_hz
from .explain import analyze_shap
from .laboratory import (
    ELECTROCHEMICAL_DEVICE, ELECTROCHEMICAL_EXPERIMENT, TASK_PATH, PlatformParameters,
    RemoteElectrochemicalParameters, bind_protocol, query_task_status, replay_task,
    submission_body, submit_to_platform,
)
from .literature import extract_with_deepseek, mine_group_pairs
from .llm import ModelClient
from .molecular import descriptor_contributions, read_catalog
from .optimization import rank_compositions
from .orchestrator import make_orchestrator
from .reflection_case import PaperReflectionCase, replay_paper_reflection
from .schemas import ExperimentResult, StartWorkflowRequest, SubmitResultsRequest


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path):
    with path.open("rb") as stream:
        return {"file": path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def _measurements(path, through_round=None):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream)
        columns = [key for key in table.fieldnames
                   if key.endswith("_pct") and key != "inhibition_efficiency_pct"]
        if len(columns) != 4:
            raise ValueError("Expected four component columns ending in _pct.")
        rows = [row for row in table if through_round is None or int(row["round"]) <= through_round]
    return (
        [[float(row[key]) for key in columns] for row in rows],
        [float(row["inhibition_efficiency_pct"]) for row in rows],
        [key.removesuffix("_pct") for key in columns],
        [row.get("source_row") for row in rows],
    )


def _spectrum(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [[float(row[key]) for key in ("frequency_hz", "real_ohm", "imag_ohm")]
                for row in csv.DictReader(stream)]


async def _demo(out, examples):
    demo_inputs = examples / "demo"
    if not demo_inputs.is_dir():
        demo_inputs = examples
    settings = Settings(coral_data_dir=out, coral_llm_mode="offline")
    chain = make_orchestrator(settings)
    ingest = chain.forgen.rag.ingest(demo_inputs / "corpus")
    record = await chain.start(StartWorkflowRequest(
        question="Design corrosion inhibitor combinations and assess adsorption geometry.", number_of_candidates=1,
    ))
    starting = record.formulations[0]
    protocol = record.protocols[0]
    remote_template = RemoteElectrochemicalParameters.model_validate(
        _read(examples / "remote_electrochemical_bindings.json"),
    )
    remote_parameters = bind_protocol(protocol, remote_template)
    local_template = PlatformParameters.model_validate(_read(examples / "platform_bindings.json"))
    local_parameters = bind_protocol(protocol, local_template)
    task_id = uuid4().hex
    identity = {"taskId": task_id, "deviceName": ELECTROCHEMICAL_DEVICE,
                "experimentName": ELECTROCHEMICAL_EXPERIMENT}
    inhibited = _spectrum(demo_inputs / "eis_inhibited.csv")
    blank = _spectrum(demo_inputs / "eis_blank.csv")

    def laboratory_service(request):
        if request.method == "POST" and request.url.path == TASK_PATH:
            if json.loads(request.content) != submission_body(remote_parameters):
                return httpx.Response(400)
            return httpx.Response(202, json={"code": "ACCEPTED", "success": True, "data": identity | {"status": 0}})
        if request.method == "GET" and request.url.path == f"{TASK_PATH}/{task_id}":
            summary = {"experimentName": ELECTROCHEMICAL_EXPERIMENT, "subType": 33, "curve": {
                "xAxis": {"name": "Z real", "unit": "ohm", "values": [row[1] for row in inhibited]},
                "yAxis": {"name": "-Z imaginary", "unit": "ohm", "values": [-row[2] for row in inhibited]},
            }}
            return httpx.Response(200, json={"code": "OK", "success": True,
                                            "data": identity | {"status": 2, "resultSummary": json.dumps(summary)}})
        return httpx.Response(404)

    async with httpx.AsyncClient(
        base_url="https://laboratory-fixture.invalid", transport=httpx.MockTransport(laboratory_service),
    ) as client:
        acceptance = await submit_to_platform(client, remote_parameters)
        dispatch = replay_task(out / "laboratory.sqlite3", local_parameters, ELECTROCHEMICAL_EXPERIMENT,
                               task_id=acceptance["taskId"])
        status = await query_task_status(client, acceptance["taskId"])
    dispatch["dispatcher_task_params"] = dispatch["request"]["body"]["experimentParams"]
    dispatch["request"] = {"path": TASK_PATH, "body": submission_body(remote_parameters)}
    dispatch["rest_receipt"] = acceptance
    dispatch["task_status_response"] = status
    dispatch["protocol_id"] = protocol.protocol_id
    dispatch["formulation_id"] = starting.formulation_id
    _write(out / "dispatch.json", dispatch)
    provenance = "Synthetic spectrum for software testing"
    eis = fit_eis(*np.asarray(inhibited).T, plot_path=out / "eis_fit.png")
    if status["status"] != 2 or dispatch["task_state"] != 2 or eis["status"] != "accepted":
        raise RuntimeError("The demonstration requires a completed task and an accepted EIS fit.")
    measured_ie = inhibition_efficiency_at_001_hz(inhibited, blank)
    eis.update({"mode": "synthetic_fixture", "derived_ie_pct": measured_ie})
    _write(out / "eis.json", eis)
    record = await chain.submit_results(record.workflow_id, SubmitResultsRequest(
        results=[ExperimentResult(formulation_id=starting.formulation_id, inhibition_efficiency=measured_ie,
                                  protocol_id=dispatch["protocol_id"], laboratory_task_id=dispatch["task_id"],
                                  source_reference=provenance,
                                  observations={"fixture": True})],
        next_step="redesign",
        substitution_pool_ids=["M0021", "M0114", "M0022"],
    ))
    trials = [ExperimentResult(
        formulation_id=item.formulation_id, inhibition_efficiency=measured_ie - i - 1,
        source_reference="Synthetic failed-substitution feedback for software testing",
        observations={"fixture": True},
    ) for i, item in enumerate(record.redesign_formulations)]
    record = await chain.submit_results(record.workflow_id, SubmitResultsRequest(results=trials, next_step="redesign"))
    formulations = {item.formulation_id: item for item in record.formulations}
    measured = {item.formulation_id: item for item in record.experiment_results}
    reflection = []
    for stage in record.mechanism_rounds:
        analysis = stage["analysis"]
        test_ids = analysis["trial_formulation_ids"]
        reflection.append({
            **analysis["reflection"], "target_group": analysis["target_group"],
            "hypothesis": stage["hypothesis"]["hypothesis"],
            "compared_trials": [{
                "formulation_id": key, "components": [item.name for item in formulations[key].components],
                "fixture_ie_pct": measured[key].inhibition_efficiency,
            } for key in test_ids],
            "candidate_comparisons": stage["screening"]["candidate_comparisons"],
            "selected_candidates": [
                chain.forgen.catalog[key]["name"] for key in stage["screening"]["selected_molecule_ids"]
            ],
        })
    _write(out / "reflection.json", {
        "mode": "synthetic_fixture",
        "paper_case_input": "examples/figure5_case_inputs.json",
        "fixture_decision_trace": reflection,
    })
    fixture_path = examples / "literature_fixture.json"
    fixture = _read(fixture_path)
    catalog = list(chain.forgen.catalog.values())
    literature = mine_group_pairs(fixture["studies"], fixture["identities"], catalog)
    literature.update({"input": _source(fixture_path), "catalog": _source(DATA / "molecular_descriptors.csv"),
                       "input_scope": fixture.get("mode", "user_supplied")})
    _write(out / "literature_pairs.json", literature)
    _write(out / "pca_group_1.json", descriptor_contributions(catalog, 1))
    compositions, efficiency, features, source_rows = _measurements(DATA / "tsc_bo.csv")
    shap = analyze_shap(compositions, efficiency, features, plot_path=out / "shap_beeswarm.png")
    shap["source_rows"] = source_rows
    shap["input"] = _source(DATA / "tsc_bo.csv")
    _write(out / "shap.json", shap)
    _write(out / "summary.json", {
        "workflow_id": record.workflow_id, "state": record.state, "rag_ingestion": ingest,
        "workflow_and_device": "Synthetic integration fixture",
        "descriptor_catalog": "Archived molecular descriptor workbook",
        "shap_data": "Supplementary Table 3",
        "reassessment_triggered": record.mechanism_rounds[-1]["analysis"]["no_improvement"],
        "next_formulation_ids": record.pending_formulation_ids,
    })
    print(out / "summary.json")


def main():
    parser = argparse.ArgumentParser(prog="python -m coral")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo")
    demo.add_argument("--out", type=Path, default=Path("coral_output/demo"))
    demo.add_argument("--examples", type=Path, default=Path("examples"))

    case = commands.add_parser("reflection-case")
    case.add_argument("--input", type=Path, default=Path("examples/figure5_case_inputs.json"))
    case.add_argument("--catalog", type=Path, default=DATA / "molecular_descriptors.csv")
    case.add_argument("--out", type=Path, default=Path("run/figure5-case.json"))

    for name in ("cluster", "pca", "mine"):
        command = commands.add_parser(name)
        if name == "cluster":
            command.add_argument("--catalog", type=Path, required=True)
            command.add_argument("--plot-dir", type=Path)
        else:
            command.add_argument("--catalog", type=Path, default=DATA / "molecular_descriptors.csv")
        command.add_argument("--out", type=Path, required=True)
        if name == "pca":
            command.add_argument("--group", type=int, required=True)
        if name == "mine":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--clusters", type=Path)
            command.add_argument("--extractor", choices=("offline", "deepseek"), default="offline")

    for name in ("bo", "shap"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=DATA / "tsc_bo.csv")
        command.add_argument("--out", type=Path, required=True)
        if name == "bo":
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--through-round", type=int, default=0)
        else:
            command.add_argument("--plot", type=Path)

    eis = commands.add_parser("fit-eis")
    eis.add_argument("--data", type=Path, required=True)
    eis.add_argument("--blank", type=Path)
    eis.add_argument("--out", type=Path, required=True)
    eis.add_argument("--plot", type=Path)

    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(_demo(args.out, args.examples))
        return
    if args.command == "reflection-case":
        source = PaperReflectionCase.model_validate(_read(args.input))
        result = replay_paper_reflection(source, read_catalog(args.catalog))
        result.update({"input": _source(args.input), "catalog": _source(args.catalog)})
        _write(args.out, result)
        print(args.out)
        return
    if getattr(args, "plot", None):
        args.plot.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "cluster":
        result = cluster_catalog(read_catalog(args.catalog), plot_dir=args.plot_dir)
    elif args.command == "pca":
        result = descriptor_contributions(read_catalog(args.catalog), args.group)
    elif args.command == "mine":
        inputs = _read(args.input)
        catalog = read_catalog(args.catalog)
        catalog_source = _source(args.catalog)
        assignments = None
        if args.clusters:
            clustering = _read(args.clusters)
            if clustering.get("input", {}).get("sha256") != catalog_source["sha256"]:
                parser.error("--clusters must be generated from the same descriptor catalog.")
            assignments = clustering["assignments"]
        extractions = None
        if args.extractor == "deepseek":
            client = ModelClient(Settings(coral_llm_mode="deepseek"))
            extractions = asyncio.run(extract_with_deepseek(inputs["studies"], client))
        result = mine_group_pairs(inputs["studies"], inputs["identities"], catalog, assignments, extractions)
        result.update({"catalog": catalog_source, "input_scope": inputs.get("mode", "user_supplied")})
        if args.clusters:
            result["cluster_input"] = _source(args.clusters)
    elif args.command == "bo":
        x, y, features, rows = _measurements(args.data, args.through_round)
        result = rank_compositions(x, y, workers=args.workers)
        result.update({"feature_order": features, "source_rows": rows, "through_round": args.through_round})
    elif args.command == "shap":
        x, y, features, rows = _measurements(args.data)
        result = analyze_shap(x, y, features, args.plot)
        result["source_rows"] = rows
    else:
        samples = _spectrum(args.data)
        result = fit_eis(*np.asarray(samples).T, plot_path=args.plot)
        if args.blank and result["status"] == "accepted":
            result["derived_ie_pct"] = inhibition_efficiency_at_001_hz(samples, _spectrum(args.blank))
    if args.command in {"cluster", "pca"}:
        input_path = args.catalog
    elif args.command == "mine":
        input_path = args.input
    else:
        input_path = args.data
    result["input"] = _source(input_path)
    if args.command == "fit-eis" and args.blank:
        result["blank_input"] = _source(args.blank)
    _write(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
