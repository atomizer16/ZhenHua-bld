from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResultPaths:
    root: str = ""
    report: str = ""
    archive: str = ""


@dataclass
class BatchProgress:
    total: int = 0
    completed: int = 0
    percent: float = 0.0
    stage: str = "idle"


@dataclass
class BatchResult:
    batch_id: str
    status: str = "created"
    error: str = ""
    progress: BatchProgress = field(default_factory=BatchProgress)
    paths: ResultPaths = field(default_factory=ResultPaths)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class ApiResponse:
    ok: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload.get("data") is None:
            payload["data"] = {}
        return payload


@dataclass
class StartScanRequest:
    mode: str  # image | video | camera
    source: str
    device: str = "cpu"
    conf_thres: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadBatchRequest:
    batch_id: str
    uploader: str  # mqtt | webdav


@dataclass
class StopScanRequest:
    batch_id: str
