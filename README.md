# CorAL

CorAL is a multi-agent self-driving laboratory for corrosion inhibitor discovery. ForGen proposes formulations from literature evidence and molecular-group priors. ExpOpt converts those formulations into electrochemical protocols and optimizes component ratios. MecIns uses experimental feedback to revise the descriptor criteria for the next candidate screen.

This repository accompanies *CorAL: A multi-agent self-driving laboratory for scientific discovery in corrosion inhibitor cocktails*. Start with the [Figure 5 reflection case](docs/reflection-case.md) to inspect the reported evidence and replay the descriptor decision. The [workflow guide](docs/workflow.md) shows how the agents, DeepSeek and the laboratory platform exchange records.

## Prerequisites

- **Local installation** uses Python 3.12, SQLite with FTS5 support, and a CPU workstation. Dependencies are declared in [`pyproject.toml`](pyproject.toml) and constrained by [`requirements-review.txt`](requirements-review.txt).
- **Offline review** uses the included example inputs and simulated equipment feedback. It requires no API key or instrument access.
- **Online model calls** use a DeepSeek API key. The application reads model settings from [`.env.example`](.env.example) when copied to `.env`.
- **Live experiments** require facility-specific instrument bindings and access to the laboratory REST service and its MQTT/PLC dispatcher. The [laboratory interface guide](docs/workflow.md#laboratory-connection) describes the handoff.

## Docker setup

Run Docker with the Compose plugin. From the directory containing this README, start the CorAL API in its default offline mode.

```sh
docker compose up --build
```

Open the [API documentation](http://127.0.0.1:18000/docs) or check [service health](http://127.0.0.1:18000/health). Compose stores workflow records and the retrieval index in the `coral_data` volume. Set `CORAL_HOST_CORPUS` to mount a local PDF directory for literature retrieval. Configure `.env` before starting the service if you want online DeepSeek calls.

The Python setup below runs the command-line demo against the examples in this repository.

## Python setup

1. Extract the archive and open a terminal in the directory containing `README.md`.
2. Create and activate a Python environment.

   ```sh
   python -m venv .venv
   ```

   On macOS or Linux, run `source .venv/bin/activate`. In PowerShell, run `.\.venv\Scripts\Activate.ps1`.

3. Install the package and test dependencies.

   ```sh
   python -m pip install -c requirements-review.txt -e ".[dev]"
   ```

   On Windows you can also use `.\.venv\Scripts\python.exe` in place of `python` to run commands with the new environment directly. Installation took about one minute on the Windows review machine with cached dependencies. Download time depends on your environment.

## Run the offline demo

The full offline demo uses a synthetic PDF and a matched pair of simulated EIS spectra in [`examples/demo/`](examples/demo/). The [saved integration traces](coral_output/README.md) can be inspected directly.

```sh
python -m coral demo --out coral_output/demo
```

The demo runs in about ten seconds on the Windows review machine. It executes the following stages.

1. **Retrieve and propose.** The demo indexes the sample PDF and asks ForGen for a corrosion inhibitor formulation under the bundled example conditions. The offline proposal is deterministic.
2. **Prepare and dispatch.** ExpOpt generates an electrochemical protocol. The laboratory adapter applies the example instrument bindings and submits a task over mock HTTP. The device replay uses the returned task ID to record the execution steps and acknowledgments.
3. **Process feedback.** The demo fits the simulated impedance spectrum, calculates inhibition efficiency against its matched simulated blank, and submits an example result to the scheduler. MecIns uses the target group's PCA descriptor ranking to define an initial LogP criterion, and ForGen screens the example shortlist.
4. **Reassess and screen again.** Synthetic substitution results fail to improve on the matched reference. MecIns retains LogP, changes its criterion to comparable hydrophobicity, and adds radius of gyration (RoG) and Labute surface area (LabuteASA). The PCA projection stays two-dimensional. ForGen applies the expanded criteria within the same group and selects DDBAB in the example.

## Expected output

Look in `coral_output/demo/`. The JSON files can be opened in a text editor.

| Output | What to inspect |
| --- | --- |
| `summary.json` | Workflow ID, final demo state and whether reassessment was triggered. |
| `reflection.json` | Baseline and trial results, criteria before and after reassessment, PCA contribution scores, candidate comparisons and the revised selection. |
| `dispatch.json` | Remote electrochemical request, acceptance and status responses, and the scheduler replay linked by task ID. |
| `workflows/<workflow_id>.json` | Full agent record with retrieved evidence, formulations, protocols, measurements and event history. |
| `eis.json` and `eis_fit.png` | Example impedance fits and calculated inhibition efficiency. |
| `literature_pairs.json`, `pca_group_1.json`, `shap.json` and `shap_beeswarm.png` | Literature pair counts, PCA descriptor contributions, and an attribution example using the archived composition data. |

Start with `fixture_decision_trace` in `reflection.json`. Read `criterion_rules_before`, `criterion_rules_after` and `selected_candidates` to follow the change from LogP alone to LogP, RoG and LabuteASA. The matching code is [`MecInsAgent.analyze`](coral/agents/mecins.py) and [`ForGenAgent.redesign`](coral/agents/forgen.py).

## Replay the Figure 5 case

```sh
python -m coral reflection-case --out run/figure5-case.json
```

The case applies the descriptor rules to values transcribed from Figure 5d and the Formulation 1 feedback displayed in Figure 5f. It records each rule result and the selected candidate. The [case guide](docs/reflection-case.md) distinguishes the reported values from the retrospective decisions.

## Run clustering and literature mining

Run the published laminarin and sodium molybdate case with the archived group assignments.

```sh
python -m coral mine --input examples/literature_fixture.json --out run/literature-pairs.json
```

To examine how recomputed clusters change the group-pair assignment, run the following commands.

```sh
python -m coral cluster --catalog coral/data/molecular_descriptors.csv --out run/clustering.json --plot-dir run/clustering
python -m coral mine --input examples/literature_fixture.json --clusters run/clustering.json --out run/literature-reclustered.json
```

The clustering command evaluates 5–15 organic clusters by silhouette score and retains the highest-scoring partition. Inorganic salts remain a predefined group. The mining command starts with prepared abstracts, conclusions and a curated identity table. It reports the source excerpt, molecular pairs, corresponding group pairs and frequency table. With the archived labels, the [published case report](coral_output/literature_pairs.json) shows one occurrence of group pair 8–11. The [scientific tool guide](docs/reproduction.md#literature-mining-and-group-frequencies) gives the input format and optional DeepSeek extraction command. The [Figure 2 data guide](docs/reproduction.md#figure-2-data) links to the archived group frequencies used by ForGen.

## Examples and study data

The [results and source data index](coral_output/README.md) links the outputs to the manuscript figures and supplementary tables.

| Location | Contents |
| --- | --- |
| [`literature_fixture.json`](examples/literature_fixture.json) | Abstract and conclusion excerpts, DOI, and catalog identities for the published laminarin and sodium molybdate case. |
| [`remote_electrochemical_bindings.json`](examples/remote_electrochemical_bindings.json) and [`platform_bindings.json`](examples/platform_bindings.json) | The public electrochemical request and the scheduler's normalized round fixture. |
| [`examples/`](examples/) | Instrument spectrum exports with a recorded 10 mV amplitude, and the [Figure 5 replay input](examples/figure5_case_inputs.json). |
| [`coral/data/`](coral/data/) | Archived molecular descriptors, group assignments and priors, and composition data from the Supplementary Information. |

See the [scientific tool guide](docs/reproduction.md) for independent PCA, SHAP, Bayesian optimization, literature mining and EIS commands.

## Online model and API usage

To enable DeepSeek, copy [`.env.example`](.env.example) to `.env` and set `CORAL_LLM_MODE=deepseek` and `CORAL_LLM_API_KEY`. The review configuration uses DeepSeek-V4-Pro with API model ID `deepseek-v4-pro`. You can set `CORAL_LLM_MODEL` to the model ID for your deployment. ForGen and MecIns use this setting when the API runs in online mode. The offline demo remains deterministic.

For a local API, start the service with the following command.

```sh
python -m uvicorn coral.main:app --host 127.0.0.1 --port 8000
```

Open [the local API documentation](http://127.0.0.1:8000/docs) to inspect the schemas and try requests. The API runs in offline mode by default. For corpus-backed proposals, set `CORAL_CORPUS_DIR` if needed and call `POST /v1/rag/ingest` before creating a workflow.

The following Python example creates a workflow and previews its first laboratory task. Use port 18000 if the service is running through Docker.

```python
import json
from pathlib import Path

import httpx

bindings = json.loads(Path("examples/remote_electrochemical_bindings.json").read_text(encoding="utf-8"))
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=200, trust_env=False) as client:
    response = client.post("/v1/workflows", json={
        "question": "Propose a corrosion inhibitor formulation for Q235 steel in 3.5 wt% NaCl.",
        "number_of_candidates": 1,
    })
    response.raise_for_status()
    run = response.json()
    workflow_id = run["workflow_id"]
    protocol_id = run["protocols"][0]["protocol_id"]
    response = client.post(
        f"/v1/workflows/{workflow_id}/protocols/{protocol_id}/dispatch-preview",
        json={"experiment_name": 204, "bindings": bindings},
    )
    response.raise_for_status()
    print(response.json()["body"])
```

The preview shows the remote electrochemical request. For live execution, the facility supplies instrument parameters to `bind_protocol()`, submits with `submit_to_platform()`, and checks the accepted task with `query_task_status()`. The [laboratory interface guide](docs/workflow.md#laboratory-connection) describes the remote payload and the facility scheduler format. Once measurements are available, `POST /v1/workflows/{workflow_id}/results` accepts the formulation ID, the protocol and laboratory task IDs for device results, and the measured inhibition efficiency. The `next_step` field selects the next agent stage after the active batch has received its results.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`.github/workflows/`](.github/workflows/) | GitHub Actions test workflow. |
| [`coral/`](coral/) | Agent scheduler, DeepSeek client, laboratory adapter, scientific tools, API and command-line interface. |
| [`coral/agents/`](coral/agents/) | ForGen, ExpOpt and MecIns. |
| [`coral/data/`](coral/data/) | Archived descriptors, priors and composition data. |
| [`coral_output/`](coral_output/) | Generated review outputs, including the Figure 5 replay, equipment trace and literature statistics. |
| [`docs/`](docs/) | Workflow and laboratory interface, plus scientific tool commands. |
| [`examples/`](examples/) | Inputs for the offline demos and scientific tool examples. |
| [`tests/`](tests/) | Tests for agent transitions, model responses, task handling and analysis tools. |

| Component | Source entry point |
| --- | --- |
| DeepSeek | [`coral/llm.py`](coral/llm.py), `ModelClient.complete_json` |
| Agent scheduling | [`coral/orchestrator.py`](coral/orchestrator.py), `start` and `submit_results` |
| Formulation and protocol | [`coral/agents/forgen.py`](coral/agents/forgen.py) and [`coral/agents/expopt.py`](coral/agents/expopt.py) |
| Equipment | [`coral/laboratory.py`](coral/laboratory.py), `bind_protocol`, `submit_to_platform`, `query_task_status` and `replay_task` |
| Reflection and rescreening | [`coral/agents/mecins.py`](coral/agents/mecins.py), `analyze`, and [`coral/agents/forgen.py`](coral/agents/forgen.py), `redesign` |
| Hierarchical clustering | [`coral/clustering.py`](coral/clustering.py), `cluster_catalog`. Run `python -m coral cluster` to examine the silhouette scores and dendrogram. |
| Literature mining and group frequencies | [`coral/literature.py`](coral/literature.py), `mine_group_pairs` and `extract_with_deepseek`. |
| Bayesian optimization | [`coral/optimization.py`](coral/optimization.py), `rank_compositions`. ExpOpt calls the same function used by the standalone BO command. |
| Other scientific tools | [`rag.py`](coral/rag.py) for retrieval, [`molecular.py`](coral/molecular.py) for PCA and descriptors, [`explain.py`](coral/explain.py) for SHAP, and [`eis.py`](coral/eis.py) for impedance analysis. |

Run `python -m pytest -q` to check the implementation in your environment.
