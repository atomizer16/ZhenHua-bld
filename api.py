from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import InferenceOrchestrator
from protocols import ApiResponse, StartScanRequest


app = FastAPI(title="Bolt Inference API", version="1.0.0")
orchestrator = InferenceOrchestrator()


class StartScanBody(BaseModel):
    mode: str = Field(pattern="^(image|video|camera)$")
    source: str
    device: str = "cpu"
    conf_thres: float = 0.7
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StopScanBody(BaseModel):
    batch_id: str


class UploadBatchBody(BaseModel):
    batch_id: str
    uploader: str = Field(pattern="^(mqtt|webdav)$")


@app.get("/health")
def health() -> Dict[str, Any]:
    return ApiResponse(ok=True, message="ok").to_dict()


@app.post("/scan/start")
def start_scan(body: StartScanBody) -> Dict[str, Any]:
    req = StartScanRequest(**body.model_dump())
    result = orchestrator.start_scan(req)
    return ApiResponse(ok=True, message="scan started", data=asdict(result)).to_dict()


@app.post("/scan/stop")
def stop_scan(body: StopScanBody) -> Dict[str, Any]:
    try:
        result = orchestrator.stop_scan(body.batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ApiResponse(ok=True, message="scan stop requested", data=asdict(result)).to_dict()


@app.get("/scan/progress/{batch_id}")
def scan_progress(batch_id: str) -> Dict[str, Any]:
    try:
        result = orchestrator.get_progress(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ApiResponse(ok=True, data=asdict(result)).to_dict()


@app.get("/scan/latest")
def latest_scan() -> Dict[str, Any]:
    result = orchestrator.latest_result()
    if result is None:
        return ApiResponse(ok=True, message="no batch yet", data={}).to_dict()
    return ApiResponse(ok=True, data=asdict(result)).to_dict()


@app.post("/scan/upload")
def upload_batch(body: UploadBatchBody) -> Dict[str, Any]:
    try:
        payload = orchestrator.upload_batch(body.batch_id, body.uploader)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApiResponse(ok=True, message="upload complete", data=payload).to_dict()
