import json
import math
import sqlite3
from contextlib import closing
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import ConfigDict, Field, model_validator

from .schemas import Record


class TestItem(Record):
    number: int = Field(ge=1)
    type: int
    parama: dict
    machineID: int | None = Field(default=None, ge=1)


class RoundSettings(Record):
    waitHour: int = Field(default=0, ge=0)
    waitMinute: int = Field(default=0, ge=0, lt=60)
    waitSecond: int = Field(default=0, ge=0, lt=60)
    testParams: list[TestItem] = Field(min_length=1)

    @property
    def wait_seconds(self):
        return self.waitHour * 3600 + self.waitMinute * 60 + self.waitSecond


class PlatformParameters(Record):
    base_setting: dict
    infos: list[RoundSettings] = Field(min_length=1)


class RemoteTestInfo(Record):
    model_config = ConfigDict(extra="forbid")

    sub_type: Literal[0, 1, 2, 21, 22, 33]
    waitHour: int = Field(default=0, ge=0)
    waitMinute: int = Field(default=0, ge=0, lt=60)
    waitSecond: int = Field(default=0, ge=0, lt=60)
    estimateHour: int = Field(default=0, ge=0)
    estimateMinute: int = Field(default=0, ge=0, lt=60)
    estimateSecond: int = Field(default=0, ge=0, lt=60)
    testParam: dict = Field(min_length=1)


class RemoteElectrochemicalParameters(Record):
    model_config = ConfigDict(extra="forbid")

    taskName: str = Field(min_length=1)
    type: Literal[4] = 4
    sampleId: int = Field(ge=1, le=12)
    sampleType: Literal[1, 2]
    machineID: Literal[1, 2]
    slowRelease: list[float] = Field(min_length=8, max_length=8)
    waterSecond: int = Field(ge=0)
    airSecond: int = Field(ge=0)
    info: RemoteTestInfo

    @model_validator(mode="after")
    def check_volume(self):
        if any(volume < 0 for volume in self.slowRelease) or sum(self.slowRelease) > 50:
            raise ValueError("Electrochemical dosing requires nonnegative volumes totaling at most 50 mL.")
        return self


class DispatchStep(Record):
    task_no: int
    round_no: int
    transport: str
    payload: dict


TRANSPORT = {
    "plc_write": "dhx/plc/write", "plc_read": "dhx/plc/read",
    "hmi_ctrl": "dhx/hmi/ctrl", "hmi_view": "dhx/hmi/view",
}
TASK_PATH = "/api/v1/experiment-tasks"
ELECTROCHEMICAL_DEVICE = 2
ELECTROCHEMICAL_EXPERIMENT = 204


def bind_protocol(protocol, bindings):
    parameters = bindings.model_dump(mode="json", exclude_none=True)
    values = {
        "formulation_id": protocol.formulation.formulation_id,
        "temperature_c": protocol.context.temperature_c,
        "eis_high_frequency_hz": protocol.eis_frequency_hz[0],
        "eis_low_frequency_hz": protocol.eis_frequency_hz[1],
        "eis_amplitude_mv": protocol.eis_amplitude_mv,
        "eis_amplitude_v": protocol.eis_amplitude_mv / 1000,
        "pdp_low_v": protocol.pdp_range_v_relative_to_ocp[0],
        "pdp_high_v": protocol.pdp_range_v_relative_to_ocp[1],
        "pdp_scan_rate_mv_per_s": protocol.pdp_scan_rate_mv_per_s,
    }
    for item in protocol.formulation.components:
        values[f"fraction:{item.molecule_id or item.name}"] = item.fraction

    def substitute(value):
        if isinstance(value, str) and value.startswith("$"):
            return values[value[1:]]
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return value

    parameters = substitute(parameters)
    first = parameters["info"] if isinstance(bindings, RemoteElectrochemicalParameters) else parameters["infos"][0]
    if not any(first.get(key, 0) for key in ("waitHour", "waitMinute", "waitSecond")):
        hours = protocol.immersion_hours
        if not float(hours).is_integer():
            raise ValueError("Supply explicit platform wait settings for fractional hours.")
        first["waitHour"] = int(hours)
    return type(bindings).model_validate(parameters)


def submission_body(parameters, experiment_name: int = ELECTROCHEMICAL_EXPERIMENT):
    if isinstance(parameters, RemoteElectrochemicalParameters) and experiment_name != ELECTROCHEMICAL_EXPERIMENT:
        raise ValueError("The remote electrochemical protocol uses experiment code 204.")
    return {"deviceName": ELECTROCHEMICAL_DEVICE, "experimentName": experiment_name,
            "experimentParams": parameters.model_dump(mode="json", exclude_none=True)}


async def submit_to_platform(client: httpx.AsyncClient, parameters, experiment_name: int = ELECTROCHEMICAL_EXPERIMENT):
    body = submission_body(parameters, experiment_name)

    def has_placeholder(value):
        if isinstance(value, dict):
            return any(has_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(has_placeholder(item) for item in value)
        return isinstance(value, str) and value.startswith("$")

    if has_placeholder(body):
        raise ValueError("Bind all protocol placeholders before submitting a laboratory task.")
    response = await client.post(TASK_PATH, json=body)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("The laboratory platform returned an invalid JSON receipt.") from error
    if not isinstance(payload, dict):
        raise ValueError("The platform did not acknowledge task creation.")
    data = payload.get("data")
    if (payload.get("code") != "ACCEPTED" or payload.get("success") is not True
            or not isinstance(data, dict) or type(data.get("status")) is not int or data["status"] != 0
            or not isinstance(data.get("taskId"), str) or not data["taskId"].strip()):
        raise ValueError("The platform did not acknowledge task creation.")
    if (data.get("deviceName", ELECTROCHEMICAL_DEVICE) != ELECTROCHEMICAL_DEVICE
            or data.get("experimentName", experiment_name) != experiment_name):
        raise ValueError("The acceptance receipt does not match the submitted experiment.")
    return data


async def query_task_status(client: httpx.AsyncClient, task_id: str):
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("A platform task ID is required.")
    response = await client.get(f"{TASK_PATH}/{quote(task_id, safe='')}")
    response.raise_for_status()
    try:
        payload = response.json()
        data = payload["data"]
        if (payload["success"] is not True or payload["code"] != "OK" or data["taskId"] != task_id
                or data["deviceName"] != ELECTROCHEMICAL_DEVICE
                or data["experimentName"] != ELECTROCHEMICAL_EXPERIMENT
                or type(data["status"]) is not int or data["status"] not in range(5)):
            raise ValueError("Mismatched task status.")
        raw_summary = data.get("resultSummary")
        summary = json.loads(raw_summary) if raw_summary is not None else None
        if raw_summary is not None:
            if not isinstance(summary, dict):
                raise ValueError("Expected a JSON object in resultSummary.")
            if summary.get("experimentName", ELECTROCHEMICAL_EXPERIMENT) != ELECTROCHEMICAL_EXPERIMENT:
                raise ValueError("Mismatched experiment in resultSummary.")
            if "curve" in summary:
                x = summary["curve"]["xAxis"]["values"]
                y = summary["curve"]["yAxis"]["values"]
                if (not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y)
                        or not all(type(point) in (int, float) and math.isfinite(point) for point in x + y)):
                    raise ValueError("Curve coordinates must be paired.")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid electrochemical task status or result summary.") from error
    return {**data, "parsed_result_summary": summary}


def dispatch_steps(task_id, parameters):
    base = parameters.base_setting
    sample = int(base["sampleId"])
    slot = int(base["storageSlotNo"])
    common = {"task_id": task_id, "sample_id": sample, "storage_slot": slot}

    def plc(number, round_no, **fields):
        payload = common | {"task_no": number, "device_cmd": number, "round_no": round_no} | fields
        return DispatchStep(task_no=number, round_no=round_no, transport="plc_write", payload=payload)

    steps = [plc(5, 0, assembly_settings=base)]
    for index, settings in enumerate(parameters.infos, 1):
        instrument = next((item.machineID for item in settings.testParams if item.machineID), 1)
        if index > 1:
            steps.append(plc(9, index, machine_id=instrument, wait_seconds=settings.wait_seconds))
        if settings.wait_seconds:
            steps.append(DispatchStep(task_no=0, round_no=index, transport="timer",
                                      payload={"wait_seconds": settings.wait_seconds}))
        steps.append(plc(6, index, machine_id=instrument))
        steps.append(DispatchStep(
            task_no=-6, round_no=index, transport="hmi_ctrl",
            payload={"task_id": task_id, "round_no": index, "machineID": instrument,
                     "test": {"id": f"{task_id}-R{index}",
                              "items": [item.model_dump(exclude_none=True) for item in settings.testParams]}},
        ))
    direct = bool(base.get("directDisassembly", False))
    steps.append(plc(8 if direct else 7, 0, **({} if direct else {"machine_id": instrument})))
    return steps


def _run_trace(task_id, steps, feedback, trace):
    def frame(channel, **expected):
        try:
            actual = next(feedback)
        except StopIteration as error:
            raise ValueError("Device feedback ended before the task completed.") from error
        trace.append(actual)
        if actual.get("channel") != channel or any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected {channel} feedback for task {task_id}.")

    def send(channel, data):
        trace.append({"channel": channel, "topic": TRANSPORT[channel], **data})

    frame("plc_read", initialized=True, automatic=True)
    for step in steps:
        if step.transport == "timer":
            trace.append({"channel": "timer", "round_no": step.round_no, **step.payload})
        elif step.transport == "hmi_ctrl":
            send("hmi_ctrl", step.payload)
            frame("hmi_view", task_id=task_id, round_no=step.round_no, machineID=step.payload["machineID"],
                  device_state=2)
        else:
            send("plc_write", step.payload)
            send("plc_write", {"rw_flag": 1})
            frame("plc_read", rw_flag=2)
            send("plc_write", {"rw_flag": 0})
            frame("plc_read", rw_flag=0)
            frame("plc_read", task_id=task_id, device_cmd=step.task_no, device_state=2)
            send("plc_write", dict.fromkeys(step.payload, 0))
    frame("plc_read", task_id=task_id, task_state=2, rw_flag=1)
    send("plc_write", {"rw_flag": 2})
    frame("plc_read", rw_flag=0)
    send("plc_write", {"rw_flag": 0})
    if next(feedback, None) is not None:
        raise ValueError("Unexpected feedback after task completion.")


def example_feedback(task_id, steps):
    yield {"channel": "plc_read", "initialized": True, "automatic": True}
    for step in steps:
        if step.transport == "plc_write":
            yield {"channel": "plc_read", "rw_flag": 2}
            yield {"channel": "plc_read", "rw_flag": 0}
            yield {"channel": "plc_read", "task_id": task_id, "device_cmd": step.task_no, "device_state": 2}
        elif step.transport == "hmi_ctrl":
            yield {"channel": "hmi_view", "task_id": task_id, "round_no": step.round_no,
                   "machineID": step.payload["machineID"], "device_state": 2}
    yield {"channel": "plc_read", "task_id": task_id, "task_state": 2, "rw_flag": 1}
    yield {"channel": "plc_read", "rw_flag": 0}


def replay_task(database, parameters, experiment_name, *, feedback=None, task_id=None):
    task_id = task_id or uuid4().hex
    request = submission_body(parameters, experiment_name)
    encoded = json.dumps(request["experimentParams"])
    with closing(sqlite3.connect(database)) as db:
        db.executescript(
            "CREATE TABLE IF NOT EXISTS lab_task (task_id TEXT PRIMARY KEY, type INTEGER, state INTEGER);"
            "CREATE TABLE IF NOT EXISTS lab_sub_task (task_id TEXT, num INTEGER, state INTEGER, task_params TEXT,"
            " PRIMARY KEY (task_id, num));"
        )
        with db:
            db.execute("INSERT INTO lab_task VALUES (?, 3, 0)", (task_id,))
            db.execute("INSERT INTO lab_sub_task VALUES (?, 1, 0, ?)", (task_id, encoded))
        task_params = db.execute("SELECT task_params FROM lab_sub_task WHERE task_id=?", (task_id,)).fetchone()[0]
        steps = dispatch_steps(task_id, PlatformParameters.model_validate_json(task_params))
        trace, error = [], None
        with db:
            db.execute("UPDATE lab_task SET state=1 WHERE task_id=?", (task_id,))
            db.execute("UPDATE lab_sub_task SET state=1 WHERE task_id=?", (task_id,))
        try:
            frames = example_feedback(task_id, steps) if feedback is None else iter(feedback)
            _run_trace(task_id, steps, frames, trace)
        except ValueError as failure:
            error = str(failure)
        state = 3 if error else 2
        with db:
            db.execute("UPDATE lab_task SET state=? WHERE task_id=?", (state, task_id))
            db.execute("UPDATE lab_sub_task SET state=? WHERE task_id=?", (state, task_id))
    return {"mode": "device_replay", "task_id": task_id, "task_state": state, "subtask_state": state,
            "request": {"path": TASK_PATH, "body": request},
            "steps": [step.model_dump() for step in steps], "trace": trace, "error": error}
