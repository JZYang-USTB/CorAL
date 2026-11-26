import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from coral.clustering import cluster_catalog
from coral.config import DATA, Settings
from coral.eis import fit_eis, synergy_index
from coral.laboratory import (
    PlatformParameters, RemoteElectrochemicalParameters, bind_protocol, dispatch_steps,
    example_feedback, query_task_status, replay_task, submit_to_platform,
)
from coral.literature import extract_with_deepseek, mine_group_pairs
from coral.llm import ModelGatewayError
from coral.main import create_app
from coral.molecular import descriptor_contributions, read_catalog
from coral.optimization import composition_grid, expected_improvement, rank_compositions
from coral.rag import RagStore
from coral.reflection_case import PaperReflectionCase, replay_paper_reflection
from coral.schemas import ExperimentProtocol


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DEMO_INPUTS = EXAMPLES / "demo"


def test_paper_reflection_case_uses_feedback_to_expand_criteria():
    inputs = json.loads((EXAMPLES / "figure5_case_inputs.json").read_text())
    catalog = read_catalog(DATA / "molecular_descriptors.csv")
    case = PaperReflectionCase.model_validate(inputs)
    result = replay_paper_reflection(case, catalog)
    assert result["initial_screen"]["selected"] == "TTAB"
    assert result["feedback"]["delta_vs_reference_pp"] == -1.1
    assert result["revised_screen"]["selected"] == "DDBAB"
    assert result["reported_follow_up"]["delta_vs_reference_pp"] == 1.9
    modified = case.model_copy(update={"follow_up": case.follow_up.model_copy(update={"inhibition_efficiency_pct": 80})})
    assert replay_paper_reflection(modified, catalog)["revised_screen"] == result["revised_screen"]
    improved = case.model_copy(update={
        "initial_feedback": case.initial_feedback.model_copy(update={"inhibition_efficiency_pct": 93}),
    })
    assert replay_paper_reflection(improved, catalog)["revised_screen"] is None


def test_workflow_pauses_dispatches_and_reassesses(tmp_path):
    client = TestClient(create_app(Settings(coral_data_dir=tmp_path, coral_llm_mode="offline")))
    response = client.post("/v1/workflows", json={"question": "corrosion inhibitor", "number_of_candidates": 1})
    assert response.status_code == 200
    started = response.json()
    assert started["state"] == "awaiting_experiment"
    assert started["experiment_results"] == []
    reference_id = started["pending_formulation_ids"][0]
    root = f"/v1/workflows/{started['workflow_id']}"
    bindings = json.loads((EXAMPLES / "platform_bindings.json").read_text())
    preview = client.post(f"{root}/protocols/{started['protocols'][0]['protocol_id']}/dispatch-preview",
                          json={"experiment_name": 1, "bindings": bindings})
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["request_path"] == "/api/v1/experiment-tasks"
    assert payload["body"]["experimentParams"]["infos"][0]["waitHour"] == 24
    assert payload["body"]["experimentParams"]["infos"][0]["testParams"][1]["parama"]["amplitudeMv"] == 5
    assert payload["formulation_id"] == reference_id
    assert client.post(root + "/results", json={"results": [{
        "formulation_id": reference_id, "protocol_id": "P-unrelated", "inhibition_efficiency": 80,
    }]}).status_code == 409
    assert client.post(root + "/results", json={"results": [{
        "formulation_id": reference_id, "laboratory_task_id": "lab-unlinked", "inhibition_efficiency": 80,
    }]}).status_code == 409
    assert "DispatchPreviewResponse" in client.get("/openapi.json").json()["components"]["schemas"]

    initial = client.post(root + "/results", json={
        "results": [{"formulation_id": reference_id, "inhibition_efficiency": 80}],
        "next_step": "redesign", "substitution_pool_ids": ["M0021", "M0114", "M0022"],
    }).json()
    assert initial["state"] == "redesign_proposed"
    assert len(initial["pending_formulation_ids"]) == 2
    p, q = initial["pending_formulation_ids"]
    assert client.post(root + "/results", json={
        "results": [{"formulation_id": reference_id, "inhibition_efficiency": 81}],
    }).status_code == 409
    partial = client.post(root + "/results", json={
        "results": [{"formulation_id": p, "inhibition_efficiency": 79}],
        "next_step": "redesign",
    }).json()
    assert len(partial["mechanism_rounds"]) == 1
    revised = client.post(root + "/results", json={
        "results": [{"formulation_id": q, "inhibition_efficiency": 78}],
        "next_step": "redesign",
    }).json()
    assert revised["mechanism_rounds"][-1]["analysis"]["no_improvement"]
    assert [item["descriptor"] for item in revised["mechanism_hypotheses"][-1]["descriptor_criteria"]] == [
        "logP", "RadiusOfGyration", "LabuteASA",
    ]
    assert revised["mechanism_rounds"][-1]["screening"]["selected_molecule_ids"] == ["M0022"]
    reflection = revised["mechanism_rounds"][-1]["analysis"]["reflection"]
    assert reflection["trigger"] == "no_improvement_vs_matched_reference"
    assert reflection["reference_ie_pct"] == 80
    assert reflection["best_trial_ie_pct"] == 79
    assert [item["descriptor"] for item in reflection["criterion_rules_before"]] == ["logP"]
    assert [item["descriptor"] for item in reflection["added_pca_descriptors"]] == [
        "RadiusOfGyration", "LabuteASA",
    ]
    assert client.get(root).json()["experiment_results"][-1]["formulation_id"] == q
    assert client.get("/v1/workflows/00000000-0000-0000-0000-000000000001").status_code == 404


def test_grid_rank_and_zero_variance(monkeypatch):
    from coral import optimization

    grid = composition_grid()
    assert grid.shape == (1771, 4)
    assert len(set(map(tuple, grid))) == len(grid)
    assert np.all(grid.sum(axis=1) == 100)
    np.testing.assert_allclose(expected_improvement([80, 90, 70], [0, 0, 0], 80), [0, 10, 0])
    calls = []

    def predict(x, y, space, seed):
        calls.append(seed)
        return np.arange(len(space), dtype=float) / len(space) * 100, {}

    monkeypatch.setattr(optimization, "_bootstrap_prediction", predict)
    ranked = rank_compositions(grid[-30:], np.full(30, 80))
    assert calls == list(range(50))
    assert ranked["cv_folds"] == 10
    assert ranked["unmeasured_grid_size"] == 1741
    assert ranked["proposals"][0]["composition_pct"] == grid[-31].tolist()


def test_ratio_seed_handoff_excludes_screen_from_training(tmp_path, monkeypatch):
    from coral.agents import expopt

    client = TestClient(create_app(Settings(coral_data_dir=tmp_path, coral_llm_mode="offline")))
    started = client.post("/v1/workflows", json={"question": "inhibitor ratios", "number_of_candidates": 1}).json()
    route = f"/v1/workflows/{started['workflow_id']}/results"
    anchor = started["pending_formulation_ids"][0]
    seeds = client.post(route, json={
        "results": [{"formulation_id": anchor, "inhibition_efficiency": 90}], "next_step": "seed_ratios",
    }).json()
    assert seeds["phase"] == "ratio_seed"
    ids = seeds["pending_formulation_ids"]
    assert len(ids) == 30

    def rank(x, y, *, excluded, workers):
        assert len(x) == len(y) == 30
        assert all(value == 80 for value in y)
        assert len(excluded) == 31
        assert [25, 25, 25, 25] in excluded
        remaining = [row.tolist() for row in composition_grid() if row.tolist() not in excluded]
        return {"proposals": [{"composition_pct": row} for row in remaining[:5]]}

    monkeypatch.setattr(expopt, "rank_compositions", rank)
    response = client.post(route, json={
        "results": [{"formulation_id": key, "inhibition_efficiency": 80} for key in ids],
        "next_step": "optimize",
    })
    record = response.json()
    assert record["phase"] == "bayesian_optimization"
    assert len(record["pending_formulation_ids"]) == 5
    assert record["optimization_rounds"][0]["training_formulation_ids"] == ids


def test_online_gateway_validates_catalog_and_citations(tmp_path, monkeypatch):
    from coral import llm

    http_client = httpx.AsyncClient
    citations = []

    def respond(request):
        assert str(request.url) == "https://fixture.invalid/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fixture"
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-pro"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["response_format"] == {"type": "json_object"}
        message = json.loads(payload["messages"][-1]["content"])
        source = message["evidence"][0]["source_id"]
        output = {"formulations": [{"components": [
            {"molecule_id": key, "fraction": 0.25} for key in ("M0008", "M0115", "M0157", "M0591")
        ], "rationale": "Fixture rationale", "source_ids": [citations[-1] if citations else source]}]}
        citations.append("invented-reference")
        return httpx.Response(200, json={
            "id": "fixture-completion", "model": "deepseek-v4-pro", "usage": {"total_tokens": 400},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
        })

    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kwargs: http_client(
        transport=httpx.MockTransport(respond), **kwargs))
    client = TestClient(create_app(Settings(
        coral_data_dir=tmp_path, coral_corpus_dir=DEMO_INPUTS / "corpus",
        coral_llm_mode="deepseek", coral_llm_base_url="https://fixture.invalid/v1",
        coral_llm_api_key="fixture",
    )))
    assert client.post("/v1/rag/ingest").json()["indexed"] == 1
    payload = {"question": "corrosion inhibitor", "number_of_candidates": 1}
    accepted = client.post("/v1/workflows", json=payload)
    assert accepted.status_code == 200
    review = accepted.json()
    assert review["formulations"][0]["source_ids"] == [review["sources"][0]["source_id"]]
    assert review["model_calls"][0]["completion_id"] == "fixture-completion"
    assert review["model_calls"][0]["usage"] == {"total_tokens": 400}
    assert review["model_calls"][0]["agent"] == "ForGen"
    assert client.post("/v1/workflows", json=payload).status_code == 502


def test_online_reflection_preserves_and_expands_pca_scope(tmp_path, monkeypatch):
    import asyncio

    from coral.llm import ModelGatewayError
    from coral.orchestrator import make_orchestrator
    from coral.schemas import ExperimentResult, StartWorkflowRequest

    async def run():
        settings = Settings(coral_data_dir=tmp_path, coral_llm_mode="offline")
        chain = make_orchestrator(settings)
        started = await chain.start(StartWorkflowRequest(question="surfactant", number_of_candidates=1))
        reference = started.formulations[0]
        baseline = ExperimentResult(formulation_id=reference.formulation_id, inhibition_efficiency=80)
        trials = [ExperimentResult(formulation_id="fixture-trial", inhibition_efficiency=79)]
        catalog = chain.forgen.catalog
        first, _, _, _ = await chain.mecins.analyze(reference, baseline, [], catalog)
        settings.coral_llm_mode = "deepseek"
        removed = True

        async def respond(_, user_message, fallback):
            content = json.loads(user_message)
            assert content["observations_failed_to_improve"]
            assert content["ranked_pca_descriptors"]
            output = content["output_shape"]
            if removed:
                output["descriptor_criteria"] = output["descriptor_criteria"][1:]
            return output, {"completion_id": "fixture-reflection"}

        monkeypatch.setattr(chain.mecins.llm, "complete_json", respond)
        with pytest.raises(ModelGatewayError):
            await chain.mecins.analyze(reference, baseline, trials, catalog, first)
        removed = False
        updated, _, summary, call = await chain.mecins.analyze(reference, baseline, trials, catalog, first)
        assert [item.descriptor for item in updated.descriptor_criteria] == [
            "logP", "RadiusOfGyration", "LabuteASA",
        ]
        assert summary["reflection"]["trigger"] == "no_improvement_vs_matched_reference"
        assert call["completion_id"] == "fixture-reflection"

    asyncio.run(run())


def test_catalog_pca_and_mining():
    catalog = read_catalog(DATA / "molecular_descriptors.csv")
    sample = [dict(row, group="") for row in catalog[:18]] + [catalog[-1]]
    cluster = cluster_catalog(sample)
    assert cluster["candidate_organic_group_counts"] == list(range(5, 16))
    scores = cluster["silhouette_scores"]
    assert list(scores) == list(range(5, 16))
    assert cluster["selected_organic_group_count"] == max(scores, key=scores.get)
    assert len(set(cluster["assignments"].values()) - {11}) == cluster["selected_organic_group_count"]
    assert cluster["adjusted_rand_against_archived_organic_labels"] is None
    assert cluster["inorganic_count"] == 1
    assert set(row["group"] for row in catalog) == {str(index) for index in range(1, 12)}
    pca = descriptor_contributions(catalog, 1)
    assert len(pca["scores"]) == 575
    assert len(pca["descriptor_contributions"]) == 15
    fixture = json.loads((EXAMPLES / "literature_fixture.json").read_text(encoding="utf-8"))
    counts = mine_group_pairs(fixture["studies"], fixture["identities"], catalog)
    assert counts["pair_frequencies"] == [{"pair": [8, 11], "frequency": 1}]
    assert counts["unresolved_or_ambiguous"] == []
    first, = counts["classified_formulations"]
    assert first["section"] == "conclusion"
    assert first["evidence"] == fixture["studies"][0]["conclusion"]
    assert first["molecular_pairs"] == [{"molecule_ids": ["M0336", "M0578"], "groups": [8, 11]}]
    groups = {row["molecule_id"]: int(row["group"]) for row in catalog}
    groups["M0336"] = 17
    updated = mine_group_pairs(fixture["studies"], fixture["identities"], catalog, assignments=groups)
    assert updated["classification_source"] == "clustering_run"
    assert updated["pair_frequencies"] == [{"pair": [11, 17], "frequency": 1}]
    fixture["studies"][0]["formulations"] = [{
        "section": "conclusion", "evidence": "A fabricated quotation.",
        "components": [{"name": "LA"}, {"name": "SM"}],
    }]
    with pytest.raises(ValueError, match="Curated evidence"):
        mine_group_pairs(fixture["studies"], fixture["identities"], catalog)


def test_mining_deepseek_checks_quote_and_preserves_unknown_names():
    evidence = "A binary inhibitor mixture of LA and a coded additive was tested."
    study = {"publication_id": "model-test", "conclusion": evidence}
    fixture = json.loads((EXAMPLES / "literature_fixture.json").read_text(encoding="utf-8"))
    catalog = read_catalog(DATA / "molecular_descriptors.csv")

    class ExtractionClient:
        async def complete_json(self, system, user, fallback):
            assert json.loads(user)["sections"] == {"conclusion": evidence}
            payload = {"section": "conclusion", "evidence": evidence, "names": self.names}
            return {"formulations": [payload]}, {"completion_id": "mock-literature"}

    client = ExtractionClient()
    client.names = ["LA", "coded additive"]
    found = asyncio.run(extract_with_deepseek([study], client))
    counts = mine_group_pairs([study], fixture["identities"], catalog, extractions=found)
    assert counts["pair_frequencies"] == []
    rejected, = counts["unresolved_or_ambiguous"]
    assert rejected["evidence"] == evidence
    assert rejected["component"] == {"name": "coded additive"}
    assert counts["model_calls"]["model-test"]["completion_id"] == "mock-literature"

    client.names = ["LA", "SM"]
    with pytest.raises(ModelGatewayError, match="Invalid extraction for model-test"):
        asyncio.run(extract_with_deepseek([study], client))


def test_pdf_retrieval_stable_ids_and_overlap(tmp_path):
    rag = RagStore(tmp_path / "index.sqlite")
    corpus = DEMO_INPUTS / "corpus"
    assert rag.ingest(corpus)["indexed"] == 1
    evidence = rag.search("Corrosion adsorption")
    assert "Synthetic retrieval fixture" in evidence[0].content
    assert evidence[0].source_id == rag.search("inhibitor")[0].source_id
    assert rag.ingest(corpus)["skipped"] == 1
    chunks = list(rag._split("a" * 1800 + "b" * 400))
    assert len(chunks) == 2
    assert chunks[0][-200:] == chunks[1][:200]


def test_device_feedback_mismatch_persists_failure(tmp_path):
    bindings = PlatformParameters.model_validate_json((EXAMPLES / "platform_bindings.json").read_text())
    steps = dispatch_steps("task-a", bindings)
    feedback = list(example_feedback("task-a", steps))
    feedback[3]["device_cmd"] = 44
    database = tmp_path / "device.sqlite"
    result = replay_task(database, bindings, 1, task_id="task-a", feedback=feedback)
    assert result["task_state"] == 3
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT state FROM lab_sub_task").fetchone() == (3,)
    assert replay_task(database, bindings, 1)["task_state"] == 2


def test_platform_rest_adapter(tmp_path):
    import asyncio

    bindings = PlatformParameters.model_validate_json((EXAMPLES / "platform_bindings.json").read_text())
    client = TestClient(create_app(Settings(coral_data_dir=tmp_path, coral_llm_mode="offline")))
    record = client.post("/v1/workflows", json={"question": "inhibitors", "number_of_candidates": 1}).json()
    protocol = ExperimentProtocol.model_validate(record["protocols"][0])
    parameters = bind_protocol(protocol, bindings)
    requests, replies = [], [
        {"code": "ACCEPTED", "success": True, "data": {"taskId": "task-a", "status": 0}},
        {"code": "ACCEPTED", "success": True, "data": None},
        {"code": "ACCEPTED", "success": True, "data": {"taskId": "task-b", "status": False}},
    ]

    def respond(request):
        assert request.url.path == "/api/v1/experiment-tasks"
        payload = json.loads(request.content)
        assert payload["deviceName"] == 2
        assert payload["experimentParams"]["infos"][0]["testParams"][1]["parama"]["amplitudeMv"] == 5
        requests.append(payload)
        return httpx.Response(200, json=replies.pop(0))

    async def run():
        async with httpx.AsyncClient(base_url="https://fixture.invalid",
                                     transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError, match="Bind all protocol placeholders"):
                await submit_to_platform(client, bindings, 1)
            assert not requests
            assert (await submit_to_platform(client, parameters, 1))["taskId"] == "task-a"
            for _ in range(2):
                with pytest.raises(ValueError, match="did not acknowledge"):
                    await submit_to_platform(client, parameters, 1)

    asyncio.run(run())


def test_remote_electrochemical_contract(tmp_path):
    import asyncio

    bindings = json.loads((EXAMPLES / "remote_electrochemical_bindings.json").read_text())
    with pytest.raises(ValueError, match="at most 50 mL"):
        RemoteElectrochemicalParameters.model_validate({**bindings, "slowRelease": [51] + [0] * 7})

    api = TestClient(create_app(Settings(coral_data_dir=tmp_path, coral_llm_mode="offline")))
    workflow = api.post("/v1/workflows", json={"question": "Electrochemical EIS", "number_of_candidates": 1}).json()
    protocol = ExperimentProtocol.model_validate(workflow["protocols"][0])
    route = f"/v1/workflows/{workflow['workflow_id']}/protocols/{protocol.protocol_id}/dispatch-preview"
    preview = api.post(route, json={"bindings": bindings})
    assert preview.status_code == 200
    body = preview.json()["body"]
    assert body["deviceName"] == 2
    assert body["experimentName"] == 204
    assert body["experimentParams"]["type"] == 4
    info = body["experimentParams"]["info"]
    assert info["sub_type"] == 33 and info["waitHour"] == 24
    assert info["testParam"]["StartFreq"] == 100000
    assert info["testParam"]["Amplitude"] == 0.005
    assert preview.json()["normalized_dispatch_steps"] == []

    parameters = bind_protocol(protocol, RemoteElectrochemicalParameters.model_validate(bindings))
    identity = {"taskId": "task-a", "deviceName": 2, "experimentName": 204}
    curve = {"xAxis": {"values": [1, 2]}, "yAxis": {"values": [3, 4]}}
    summaries = [None, json.dumps({"curve": curve}), json.dumps({"curve": {**curve, "yAxis": {"values": [3]}}})]

    def respond(request):
        if request.method == "POST":
            assert request.url.path == "/api/v1/experiment-tasks"
            assert json.loads(request.content) == body
            return httpx.Response(202, json={"code": "ACCEPTED", "success": True, "data": identity | {"status": 0}})
        assert request.url.path == "/api/v1/experiment-tasks/task-a"
        summary = summaries.pop(0)
        return httpx.Response(200, json={"code": "OK", "success": True,
                                        "data": identity | {"status": 2 if summary else 1,
                                                            "resultSummary": summary}})

    async def run():
        async with httpx.AsyncClient(base_url="https://fixture.invalid", transport=httpx.MockTransport(respond)) as client:
            receipt = await submit_to_platform(client, parameters)
            task_id = receipt["taskId"]
            assert (await query_task_status(client, task_id))["parsed_result_summary"] is None
            assert (await query_task_status(client, task_id))["parsed_result_summary"]["curve"] == curve
            with pytest.raises(ValueError, match="result summary"):
                await query_task_status(client, task_id)

    asyncio.run(run())


def test_eis_threshold_and_synergy():
    spectrum = np.genfromtxt(DEMO_INPUTS / "eis_inhibited.csv", delimiter=",", names=True)
    columns = [spectrum[name] for name in spectrum.dtype.names]
    fitted = fit_eis(*columns)
    assert fitted["status"] == "accepted"
    assert len(fitted["attempts"]) == 3
    failed = fit_eis(*columns, threshold=1e-30)
    assert failed["status"] == "remeasurement_required"
    assert len(failed["attempts"]) == 4
    assert synergy_index(75, [0.5, 0.5], [20, 40])["synergy_index_pct"] == 150
    assert synergy_index(75, [0.5, 0.5], [-20, 0])["synergy_index_pct"] is None
