from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import py7zr

from protocols import BatchProgress, BatchResult, ResultPaths, StartScanRequest


def read_device_id(default: str = "UNKNOWN-DEVICE") -> str:
    return os.getenv("DEVICE_ID", default)


def read_camera_serials(mapping_file: str = "camera_position_map.json") -> Dict[str, str]:
    path = Path(mapping_file)
    if not path.exists():
        return {"left": "", "right": ""}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"left": "", "right": ""}

    serials = {"left": "", "right": ""}
    if isinstance(payload, dict):
        for side in ("left", "right"):
            value = payload.get(side, "")
            serials[side] = str(value or "")
    return serials


def normalize_bolt_id(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(int(value))
    text = str(value or "").strip()
    m = re.search(r"\d+", text)
    return m.group(0) if m else text


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_bolt: Dict[str, Dict[str, Any]] = {}
    for record in records:
        bolt_id = normalize_bolt_id(record.get("bolt_id"))
        if not bolt_id:
            continue
        current = by_bolt.get(bolt_id)
        if current is None or float(record.get("confidence", 0.0)) >= float(current.get("confidence", 0.0)):
            by_bolt[bolt_id] = record
    return by_bolt


def write_report(records: List[Dict[str, Any]], report_path: str) -> str:
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if report_path.endswith(".xlsx"):
        df.to_excel(report_path, index=False)
    else:
        df.to_csv(report_path, index=False)
    return report_path


def compress_7z(scan_dir: str, status_callback: Optional[Callable[[str], None]] = None) -> str:
    if not scan_dir or not os.path.isdir(scan_dir):
        raise FileNotFoundError(f"scan dir not found: {scan_dir}")
    if callable(status_callback):
        status_callback("compressing")
    batch_name = os.path.basename(os.path.normpath(scan_dir))
    archive_path = os.path.join(os.path.dirname(scan_dir), f"{batch_name}.7z")
    filters = [{"id": py7zr.FILTER_LZMA2, "preset": 7}]
    with py7zr.SevenZipFile(archive_path, mode="w", filters=filters) as zf:
        zf.writeall(scan_dir, arcname=batch_name)
    if callable(status_callback):
        status_callback("compressed")
    return archive_path


class MqttUploader:
    def upload_batch(self, archive_path: str, status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if callable(status_callback):
            status_callback("uploading")
        # 真实实现可接入 paho-mqtt；这里保留统一调度入口
        time.sleep(0.2)
        if callable(status_callback):
            status_callback("uploaded")
        return {"uploader": "mqtt", "archive": archive_path}


class WebDAVUploader:
    def upload_batch(self, archive_path: str, status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if callable(status_callback):
            status_callback("uploading")
        # 真实实现可接入 webdavclient3；这里保留统一调度入口
        time.sleep(0.2)
        if callable(status_callback):
            status_callback("uploaded")
        return {"uploader": "webdav", "archive": archive_path}


class InferenceOrchestrator:
    """推理编排层：集中处理批次状态、任务调度、结果聚合、报表与上传。"""

    def __init__(self, result_root: str = "runs/refactor_batches"):
        self.result_root = Path(result_root)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop_flags: Dict[str, threading.Event] = {}
        self._results: Dict[str, BatchResult] = {}
        self._latest_batch_id: Optional[str] = None
        self._uploaders = {
            "mqtt": MqttUploader(),
            "webdav": WebDAVUploader(),
        }

    def start_scan(self, req: StartScanRequest) -> BatchResult:
        batch_id = uuid.uuid4().hex[:12]
        batch_dir = self.result_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        result = BatchResult(
            batch_id=batch_id,
            status="running",
            progress=BatchProgress(total=100, completed=0, percent=0.0, stage=f"{req.mode}:queued"),
            paths=ResultPaths(root=str(batch_dir)),
            summary={
                "mode": req.mode,
                "source": req.source,
                "device": req.device,
                "conf_thres": req.conf_thres,
                "device_id": read_device_id(),
                "camera_serials": read_camera_serials(),
            },
        )
        stop_event = threading.Event()
        with self._lock:
            self._results[batch_id] = result
            self._stop_flags[batch_id] = stop_event
            self._latest_batch_id = batch_id

        thread = threading.Thread(target=self._run_scan_job, args=(batch_id, req, stop_event), daemon=True)
        thread.start()
        return result

    def stop_scan(self, batch_id: str) -> BatchResult:
        with self._lock:
            stop_event = self._stop_flags.get(batch_id)
            result = self._results.get(batch_id)
        if result is None:
            raise KeyError(f"unknown batch id: {batch_id}")
        if stop_event is not None:
            stop_event.set()
        return result

    def get_progress(self, batch_id: str) -> BatchResult:
        with self._lock:
            result = self._results.get(batch_id)
        if result is None:
            raise KeyError(f"unknown batch id: {batch_id}")
        return result

    def latest_result(self) -> Optional[BatchResult]:
        with self._lock:
            if not self._latest_batch_id:
                return None
            return self._results.get(self._latest_batch_id)

    def upload_batch(self, batch_id: str, uploader_name: str) -> Dict[str, Any]:
        with self._lock:
            result = self._results.get(batch_id)
        if result is None:
            raise KeyError(f"unknown batch id: {batch_id}")
        uploader = self._uploaders.get(uploader_name)
        if uploader is None:
            raise ValueError(f"unsupported uploader: {uploader_name}")
        archive_path = result.paths.archive
        if not archive_path:
            archive_path = compress_7z(result.paths.root)
            result.paths.archive = archive_path
        info = uploader.upload_batch(archive_path)
        result.summary["upload"] = info
        return info

    def _run_scan_job(self, batch_id: str, req: StartScanRequest, stop_event: threading.Event) -> None:
        simulated_records: List[Dict[str, Any]] = []
        for i in range(1, 101):
            if stop_event.is_set():
                self._update_result(batch_id, status="stopped", stage="stopped", completed=i - 1)
                return
            simulated_records.append(
                {
                    "image_id": f"{req.mode}_{i:04d}",
                    "bolt_id": f"B{i%10}",
                    "status": "loose" if i % 17 == 0 else "normal",
                    "confidence": round(0.4 + (i % 60) / 100.0, 3),
                }
            )
            self._update_result(batch_id, status="running", stage=f"{req.mode}:infer", completed=i)
            time.sleep(0.01)

        aggregated = aggregate_records(simulated_records)
        report_path = str(self.result_root / batch_id / "report.csv")
        write_report(list(aggregated.values()), report_path)
        archive_path = compress_7z(str(self.result_root / batch_id))
        self._update_result(
            batch_id,
            status="finished",
            stage="finished",
            completed=100,
            paths={"report": report_path, "archive": archive_path},
            summary={"record_count": len(simulated_records), "aggregated_count": len(aggregated)},
        )

    def _update_result(
        self,
        batch_id: str,
        *,
        status: str,
        stage: str,
        completed: int,
        paths: Optional[Dict[str, str]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            result = self._results[batch_id]
            result.status = status
            result.progress.completed = completed
            result.progress.total = 100
            result.progress.percent = round((completed / 100.0) * 100.0, 2)
            result.progress.stage = stage
            result.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if paths:
                for k, v in paths.items():
                    setattr(result.paths, k, v)
            if summary:
                result.summary.update(summary)

    @staticmethod
    def as_dict(result: BatchResult) -> Dict[str, Any]:
        return asdict(result)
