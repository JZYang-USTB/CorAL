import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from .schemas import WorkflowRecord


class WorkflowStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, workflow_id: str) -> Path:
        try:
            key = UUID(workflow_id)
        except (ValueError, AttributeError) as error:
            raise FileNotFoundError(workflow_id) from error
        return self.directory / f"{key}.json"

    def save(self, record: WorkflowRecord) -> WorkflowRecord:
        record.updated_at = datetime.now(timezone.utc)
        target = self._path(record.workflow_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)
        return record

    def load(self, workflow_id: str) -> WorkflowRecord:
        target = self._path(workflow_id)
        if not target.exists():
            raise FileNotFoundError(workflow_id)
        return WorkflowRecord.model_validate_json(target.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        records = []
        for path in sorted(self.directory.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append({
                    "workflow_id": payload["workflow_id"],
                    "state": payload["state"],
                    "created_at": payload["created_at"],
                    "updated_at": payload["updated_at"],
                    "question": payload["request"]["question"],
                })
            except (KeyError, json.JSONDecodeError):
                continue
        return records

