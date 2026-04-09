from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

import requests
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        resp = requests.request(method, f"{self.base_url}{path}", timeout=8, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def start_scan(self, mode: str, source: str, device: str, conf_thres: float) -> Dict[str, Any]:
        return self._request("POST", "/scan/start", json={"mode": mode, "source": source, "device": device, "conf_thres": conf_thres})

    def stop_scan(self, batch_id: str) -> Dict[str, Any]:
        return self._request("POST", "/scan/stop", json={"batch_id": batch_id})

    def get_progress(self, batch_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/scan/progress/{batch_id}")

    def latest_scan(self) -> Dict[str, Any]:
        return self._request("GET", "/scan/latest")

    def upload_batch(self, batch_id: str, uploader: str) -> Dict[str, Any]:
        return self._request("POST", "/scan/upload", json={"batch_id": batch_id, "uploader": uploader})


class FunctionPage(QWidget):
    mode = "image"

    def __init__(self, mw: "MainWindow", title: str):
        super().__init__()
        self.mw = mw
        self.batch_id: str = ""
        self._build_ui(title)

    def _build_ui(self, title: str) -> None:
        root = QVBoxLayout(self)

        root.addWidget(QLabel(f"<h2>{title}</h2>"))

        form = QFormLayout()
        self.source_input = QLineEdit()
        self.device_input = QLineEdit("cpu")
        self.conf_input = QLineEdit("0.7")
        form.addRow("输入源", self.source_input)
        form.addRow("设备", self.device_input)
        form.addRow("置信度", self.conf_input)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始扫描")
        self.btn_stop = QPushButton("停止扫描")
        self.btn_upload_mqtt = QPushButton("上传 MQTT")
        self.btn_upload_webdav = QPushButton("上传 WebDAV")
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_upload_mqtt)
        btn_row.addWidget(self.btn_upload_webdav)
        root.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        root.addWidget(QLabel("预览区（后续可接图片/视频帧预览）"))
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        root.addWidget(self.preview)

        self.btn_start.clicked.connect(self._start_scan)
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_upload_mqtt.clicked.connect(lambda: self._upload("mqtt"))
        self.btn_upload_webdav.clicked.connect(lambda: self._upload("webdav"))

    def _start_scan(self) -> None:
        try:
            payload = self.mw.api.start_scan(
                mode=self.mode,
                source=self.source_input.text().strip(),
                device=self.device_input.text().strip() or "cpu",
                conf_thres=float(self.conf_input.text().strip() or "0.7"),
            )
            data = payload.get("data", {})
            self.batch_id = data.get("batch_id", "")
            self.preview.appendPlainText(f"已开始: {self.batch_id}")
        except Exception as exc:
            QMessageBox.critical(self, "调用失败", str(exc))

    def _stop_scan(self) -> None:
        if not self.batch_id:
            return
        try:
            self.mw.api.stop_scan(self.batch_id)
            self.preview.appendPlainText(f"停止请求已发送: {self.batch_id}")
        except Exception as exc:
            QMessageBox.critical(self, "调用失败", str(exc))

    def _upload(self, uploader: str) -> None:
        if not self.batch_id:
            return
        try:
            payload = self.mw.api.upload_batch(self.batch_id, uploader)
            self.preview.appendPlainText(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            QMessageBox.critical(self, "上传失败", str(exc))

    def refresh_progress(self) -> None:
        if not self.batch_id:
            return
        try:
            payload = self.mw.api.get_progress(self.batch_id)
            data = payload.get("data", {})
            p = data.get("progress", {})
            pct = int(p.get("percent", 0) or 0)
            self.progress.setValue(max(0, min(100, pct)))
            self.preview.appendPlainText(
                f"[{data.get('batch_id')}] {data.get('status')} | {p.get('stage')} | {p.get('percent')}% | {data.get('paths', {}).get('report', '')}"
            )
        except Exception:
            # 轮询场景不打断 UI
            pass


class ImageInferencePage(FunctionPage):
    mode = "image"

    def __init__(self, mw: "MainWindow"):
        super().__init__(mw, "图片检测")


class VideoInferencePage(FunctionPage):
    mode = "video"

    def __init__(self, mw: "MainWindow"):
        super().__init__(mw, "视频检测")


class CameraPage(FunctionPage):
    mode = "camera"

    def __init__(self, mw: "MainWindow"):
        super().__init__(mw, "摄像头检测")


class MainWindow(QMainWindow):
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.setWindowTitle("岸桥轨道螺栓松动监测系统（UI/API 解耦版）")
        self.resize(1200, 800)
        self.api = ApiClient(api_base_url)

        self.stacked = QStackedWidget()
        self.page_image = ImageInferencePage(self)
        self.page_video = VideoInferencePage(self)
        self.page_camera = CameraPage(self)
        for page in (self.page_image, self.page_video, self.page_camera):
            self.stacked.addWidget(page)
        self.setCentralWidget(self.stacked)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1200)

    def _poll(self) -> None:
        for page in (self.page_image, self.page_video, self.page_camera):
            page.refresh_progress()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
