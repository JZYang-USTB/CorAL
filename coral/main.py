from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .config import Settings
from .laboratory import (
    ELECTROCHEMICAL_EXPERIMENT, TASK_PATH, DispatchStep, PlatformParameters,
    RemoteElectrochemicalParameters, bind_protocol, dispatch_steps, submission_body,
)
from .llm import ModelGatewayError
from .orchestrator import make_orchestrator
from .schemas import Record, StartWorkflowRequest, SubmitResultsRequest, WorkflowRecord


class DispatchPreviewRequest(Record):
    experiment_name: int = ELECTROCHEMICAL_EXPERIMENT
    bindings: RemoteElectrochemicalParameters | PlatformParameters


class DispatchPreviewResponse(Record):
    protocol_id: str
    formulation_id: str
    request_path: str
    body: dict
    normalized_dispatch_steps: list[DispatchStep]


def create_app(settings=None):
    settings = settings or Settings()
    chain = make_orchestrator(settings)
    rag = chain.forgen.rag
    service = FastAPI(title="CorAL reviewer implementation", version="1.7.1")

    @service.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/docs")

    @service.get("/health")
    def health():
        return {"status": "ok", "model_mode": settings.coral_llm_mode, "retrieval": rag.status()}

    @service.post("/v1/rag/ingest")
    def ingest(limit: int | None = Query(default=None, ge=1)):
        return rag.ingest(settings.coral_corpus_dir, limit)

    @service.get("/v1/rag/status")
    def retrieval_status():
        return rag.status()

    @service.post("/v1/workflows", response_model=WorkflowRecord)
    async def start(request: StartWorkflowRequest):
        try:
            return await chain.start(request)
        except ModelGatewayError as error:
            raise HTTPException(502, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @service.get("/v1/workflows")
    def list_workflows():
        return chain.store.list()

    @service.get("/v1/workflows/{workflow_id}", response_model=WorkflowRecord)
    def get_workflow(workflow_id: UUID):
        try:
            return chain.store.load(str(workflow_id))
        except FileNotFoundError as error:
            raise HTTPException(404, "Workflow not found.") from error

    @service.post("/v1/workflows/{workflow_id}/results", response_model=WorkflowRecord)
    async def submit_results(workflow_id: UUID, request: SubmitResultsRequest):
        try:
            return await chain.submit_results(str(workflow_id), request)
        except FileNotFoundError as error:
            raise HTTPException(404, "Workflow not found.") from error
        except ModelGatewayError as error:
            raise HTTPException(502, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @service.post("/v1/workflows/{workflow_id}/protocols/{protocol_id}/dispatch-preview",
                  response_model=DispatchPreviewResponse)
    def preview_dispatch(workflow_id: UUID, protocol_id: str, request: DispatchPreviewRequest):
        record = get_workflow(workflow_id)
        protocol = next((item for item in record.protocols if item.protocol_id == protocol_id), None)
        if protocol is None:
            raise HTTPException(404, "Protocol not found.")
        try:
            parameters = bind_protocol(protocol, request.bindings)
            body = submission_body(parameters, request.experiment_name)
            steps = dispatch_steps(protocol_id, parameters) if isinstance(parameters, PlatformParameters) else []
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(422, "Invalid platform bindings.") from error
        return {"protocol_id": protocol_id, "formulation_id": protocol.formulation.formulation_id,
                "request_path": TASK_PATH,
                "body": body,
                "normalized_dispatch_steps": steps}

    return service


app = create_app()
