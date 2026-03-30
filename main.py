#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import warnings
import hashlib
import base64
import io
import re
import datetime
import math
import subprocess
import platform
import uuid
from pathlib import Path
import cv2
from PIL import Image
import shutil
import pandas as pd
import paho.mqtt.client as mqtt
from webdav3.client import Client
import requests
import json
from urllib.parse import urljoin, urlparse, urlunparse, unquote
from xml.etree import ElementTree

# —— 新增：动态获取应用根目录 ——
if getattr(sys, "frozen", False):
    # PyInstaller 打包后，资源临时解压到 _MEIPASS
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 用户数据文件路径
USER_DATA_FILE = os.path.join(BASE_DIR, "users.json")
CAMERA_POSITION_MAP_FILE = os.path.join(BASE_DIR, "camera_position_map.json")


def hash_password(password):
    """生成密码哈希"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def load_users():
    """加载用户数据，返回字典 {username: password_hash}"""
    import json
    if not os.path.exists(USER_DATA_FILE):
        return {}
    try:
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            if isinstance(users, dict):
                return users
            else:
                return {}
    except Exception as e:
        return {}

def save_users(users):
    """保存用户数据字典到文件"""
    import json
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)


# —— 跟踪器状态管理 ——
def reset_tracker_state(model):
    """重置 YOLO 跟踪器的内部计数，避免不同功能之间 ID 交叉污染。

    Video 和 图片批量检测共用同一个模型实例，为确保每次调用 track()
    时的 ID 都从 1 开始，需要在真正开始推理前清空跟踪器的缓存。
    """
    try:
        tracker = getattr(model, "tracker", None)
        if tracker is None:
            return

        # Ultralytics 内部的主 tracker
        inner_tracker = getattr(tracker, "tracker", None)
        if inner_tracker and hasattr(inner_tracker, "reset"):
            inner_tracker.reset()

        # 某些实现里会有 tracker_list（多源）
        tracker_list = getattr(tracker, "tracker_list", None)
        if tracker_list:
            for t in tracker_list:
                if hasattr(t, "reset"):
                    t.reset()
    except Exception:
        # 重置失败不阻塞主流程
        pass



def is_loose_status(status):
    return str(status or "").strip().lower() == "loose"


def bolt_id_sort_key(value):
    if isinstance(value, (int, float)):
        return (0, float(value), str(value))
    text = str(value)
    m = re.search(r"\d+", text)
    if m:
        return (0, float(m.group()), text)
    return (1, float("inf"), text)


def _candidate_sort_key(record):
    return (
        1 if is_loose_status(record.get("status") or record.get("class")) else 0,
        float(record.get("confidence", record.get("conf", 0.0)) or 0.0),
    )


def pick_better_bolt_record(current, candidate):
    if current is None:
        return candidate
    return candidate if _candidate_sort_key(candidate) > _candidate_sort_key(current) else current


def aggregate_bolt_records(records):
    aggregated = {}
    for record in records or []:
        bolt_id = record.get("bolt_id")
        if bolt_id is None:
            continue
        normalized = dict(record)
        if "status" not in normalized and "class" in normalized:
            normalized["status"] = normalized.get("class")
        if "class" not in normalized and "status" in normalized:
            normalized["class"] = normalized.get("status")
        if "confidence" not in normalized and "conf" in normalized:
            normalized["confidence"] = normalized.get("conf")
        if "conf" not in normalized and "confidence" in normalized:
            normalized["conf"] = normalized.get("confidence")
        aggregated[bolt_id] = pick_better_bolt_record(aggregated.get(bolt_id), normalized)
    return aggregated


UNIFIED_REPORT_COLUMNS = ["图片ID", "螺栓ID", "螺栓状态", "置信度", "x1", "y1", "x2", "y2"]


def to_unified_report_rows(records):
    unified_rows = []
    for record in records or []:
        unified_rows.append(
            {
                "图片ID": record.get("图片ID", ""),
                "螺栓ID": record.get("螺栓ID", record.get("bolt_id", "")),
                "螺栓状态": record.get("螺栓状态", record.get("status", record.get("class", ""))),
                "置信度": record.get("置信度", record.get("confidence", record.get("conf", 0.0))),
                "x1": record.get("x1", ""),
                "y1": record.get("y1", ""),
                "x2": record.get("x2", ""),
                "y2": record.get("y2", ""),
            }
        )
    return unified_rows


def to_unified_report_df(records):
    return pd.DataFrame(to_unified_report_rows(records), columns=UNIFIED_REPORT_COLUMNS)


def aggregated_records_to_rows(aggregated, mode="image"):
    rows = []
    for bolt_id, record in sorted(aggregated.items(), key=lambda item: bolt_id_sort_key(item[0])):
        if mode == "video":
            rows.append({
                "图片ID": record.get("图片ID", ""),
                "frame": record.get("frame", ""),
                "bolt_id": bolt_id,
                "status": record.get("status", ""),
                "conf": record.get("conf", record.get("confidence", 0.0)),
                "x1": record.get("x1", ""),
                "y1": record.get("y1", ""),
                "x2": record.get("x2", ""),
                "y2": record.get("y2", ""),
                "raw_path": record.get("raw_path", ""),
                "det_path": record.get("det_path", ""),
            })
        else:
            rows.append({
                "图片ID": record.get("图片ID", ""),
                "bolt_id": bolt_id,
                "class": record.get("class", record.get("status", "")),
                "confidence": record.get("confidence", record.get("conf", 0.0)),
                "x1": record.get("x1", ""),
                "y1": record.get("y1", ""),
                "x2": record.get("x2", ""),
                "y2": record.get("y2", ""),
                "raw_path": record.get("raw_path", ""),
                "det_path": record.get("det_path", ""),
            })
    return rows

def _read_command_output(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except Exception:
        return ""


def get_motherboard_serial():
    """按平台读取主板序列号，失败时返回带原因的 fallback 标志。"""
    system_name = platform.system().lower()
    candidates = []

    if system_name == "windows":
        output = _read_command_output(["wmic", "baseboard", "get", "serialnumber"])
        for line in output.splitlines():
            value = line.strip()
            if value and value.lower() != "serialnumber":
                candidates.append(value)
    elif system_name == "linux":
        for serial_path in (
            "/sys/class/dmi/id/board_serial",
            "/sys/devices/virtual/dmi/id/board_serial",
            "/sys/class/dmi/id/product_serial",
        ):
            try:
                if os.path.exists(serial_path):
                    value = Path(serial_path).read_text(encoding="utf-8", errors="ignore").strip()
                    if value:
                        candidates.append(value)
            except Exception:
                continue
        if not candidates:
            output = _read_command_output(["dmidecode", "-s", "baseboard-serial-number"])
            if output:
                candidates.extend([line.strip() for line in output.splitlines() if line.strip()])
    elif system_name == "darwin":
        output = _read_command_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        match = re.search(r'"IOPlatformSerialNumber"\s*=\s*"([^"]+)"', output)
        if match:
            candidates.append(match.group(1).strip())

    invalid_values = {"", "none", "unknown", "system serial number", "to be filled by o.e.m.", "default string"}
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized.lower() not in invalid_values:
            return normalized

    fallback_id = hex(uuid.getnode())[2:].upper() or "UNKNOWN"
    return f"FALLBACK_UNAVAILABLE_{system_name.upper()}_{fallback_id}"


def load_camera_position_map():
    if not os.path.exists(CAMERA_POSITION_MAP_FILE):
        return {}
    try:
        with open(CAMERA_POSITION_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if str(v) in {"left", "right"}}
    except Exception:
        pass
    return {}


def save_camera_position_map(mapping):
    normalized = {str(k): str(v) for k, v in mapping.items() if str(v) in {"left", "right"}}
    with open(CAMERA_POSITION_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def _sanitize_camera_token(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return value.strip("_")


def _normalize_camera_serial(value):
    value = str(value or "").strip()
    invalid_values = {"", "none", "unknown", "n/a", "null"}
    if value.lower() in invalid_values:
        return ""
    return value


def _read_device_properties_linux(device_path):
    output = _read_command_output(["udevadm", "info", "--query=property", "--name", device_path])
    properties = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key.strip()] = value.strip()
    return properties


def get_camera_metadata(camera_index):
    metadata = {
        "camera_index": camera_index,
        "camera_serial": "",
        "vendor": "Generic",
        "model": f"Camera {camera_index}",
        "display_name": f"Camera {camera_index}",
        "serial_source": "unavailable",
        "is_limited_fallback": True,
        "warning": "未读取到稳定序列号，仅适合临时调试，不适合正式归档。",
    }

    system_name = platform.system().lower()
    if system_name == "linux":
        device_path = f"/dev/video{camera_index}"
        props = _read_device_properties_linux(device_path)
        vendor = props.get("ID_VENDOR_FROM_DATABASE") or props.get("ID_VENDOR") or props.get("ID_VENDOR_ID") or metadata["vendor"]
        model = props.get("ID_MODEL_FROM_DATABASE") or props.get("ID_MODEL") or metadata["model"]
        serial = _normalize_camera_serial(props.get("ID_SERIAL_SHORT") or props.get("ID_SERIAL"))
        metadata.update({
            "vendor": vendor,
            "model": model,
            "display_name": f"{vendor} {model}".strip(),
        })
        if serial:
            metadata["camera_serial"] = serial
            metadata["serial_source"] = "udevadm"
            metadata["is_limited_fallback"] = False
            metadata["warning"] = ""
    elif system_name == "windows":
        metadata.update({
            "vendor": "USB",
            "model": f"USB Camera {camera_index}",
            "display_name": f"USB Camera {camera_index}",
            "warning": "当前环境未提供稳定的 Windows 相机序列号读取能力，将使用受限 fallback，不适合正式归档。",
        })
    else:
        metadata["warning"] = f"当前平台（{system_name}）未实现稳定序列号读取，将使用受限 fallback，不适合正式归档。"

    if metadata["camera_serial"]:
        metadata["camera_identity"] = metadata["camera_serial"]
    else:
        fallback_key = "USB_FALLBACK_{index}_{vendor}_{model}".format(
            index=camera_index,
            vendor=_sanitize_camera_token(metadata.get("vendor", "generic")),
            model=_sanitize_camera_token(metadata.get("model", f"camera_{camera_index}")),
        )
        metadata["camera_identity"] = fallback_key
    return metadata


def open_camera_device(preferred_index=None, max_indices=5):
    indices = []
    if preferred_index is not None:
        indices.append(int(preferred_index))
    indices.extend(idx for idx in range(max_indices) if idx not in indices)

    last_error = ""
    for camera_index in indices:
        try:
            cap = cv2.VideoCapture(camera_index)
        except Exception as exc:
            last_error = str(exc)
            continue
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue

        metadata = get_camera_metadata(camera_index)
        metadata["backend_name"] = getattr(cap, "getBackendName", lambda: "unknown")()
        return {
            "capture": cap,
            "camera_index": camera_index,
            "camera_serial": metadata.get("camera_serial", ""),
            "vendor": metadata.get("vendor", "Generic"),
            "model": metadata.get("model", f"Camera {camera_index}"),
            "camera_identity": metadata.get("camera_identity", ""),
            "metadata": metadata,
        }

    raise RuntimeError(last_error or "无法打开摄像头，请检查设备连接。")


def build_scan_metadata(scan_type, extra_metadata=None):
    serial = get_motherboard_serial()
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    metadata = {
        "device_id": serial,
        "motherboard_serial": serial,
        "scan_time": timestamp,
        "scan_type": scan_type,
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata[key] = value
    return metadata


def write_id_file(scan_dir_or_text_part, metadata):
    target_dir = scan_dir_or_text_part
    if os.path.basename(os.path.normpath(scan_dir_or_text_part)) != "text_part":
        target_dir = os.path.join(scan_dir_or_text_part, "text_part")
    os.makedirs(target_dir, exist_ok=True)
    id_path = os.path.join(target_dir, "ID.TXT")
    lines = []
    for key in ("device_id", "motherboard_serial", "scan_time", "scan_type"):
        value = metadata.get(key, "")
        lines.append(f"{key}={value}")
    for key, value in metadata.items():
        if key not in {"device_id", "motherboard_serial", "scan_time", "scan_type"}:
            lines.append(f"{key}={value}")
    with open(id_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return id_path


def refresh_scan_id(scan_layout, scan_type, extra_metadata=None):
    metadata = build_scan_metadata(scan_type, extra_metadata=extra_metadata)
    write_id_file(scan_layout.get("text_part") or scan_layout.get("scan_dir") or "", metadata)
    return metadata


# —— 资源路径统一管理 ——
CRANE_IMAGE_PATH = os.path.join(BASE_DIR, "crane.jpg")
WEIGHTS_PATH     = os.path.join(BASE_DIR, "best.pt")
BOTSORT_CONFIG   = os.path.join(BASE_DIR, "botsort.yaml")
# ——————————————————————————————

# 设置中文字体，减少 missing glyph 警告（需系统安装 SimSun 等字体）
import matplotlib
matplotlib.rc("font", family="SimSun", size=11)
matplotlib.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np
import pandas as pd
import joblib
# (禁用Agg后，使用Qt5Agg以支持交互绘图)
# matplotlib.use("Agg")  # 移除Agg，使用默认交互后端
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QImage, QFont, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit,
    QSlider, QLineEdit, QProgressBar, QFrame, QGridLayout,
    QStackedWidget, QDialog, QSizePolicy, QProgressDialog, QScrollArea
)

import cv2
from ultralytics import YOLO
from PIL import Image


###############################################################################
#   统一的界面视觉与交互规范
###############################################################################
class UITheme:
    """集中管理界面配色、字体与空间规范。"""

    FONT_FAMILY = "Microsoft YaHei"
    FONT_SIZE_BASE = 12
    FONT_SIZE_SUBTITLE = 14
    FONT_SIZE_TITLE = 18

    COLOR_BACKGROUND = "#F3F6FC"
    COLOR_SURFACE = "#FFFFFF"
    COLOR_PRIMARY = "#2563EB"
    COLOR_PRIMARY_HOVER = "#1D4ED8"
    COLOR_PRIMARY_PRESSED = "#1E40AF"
    COLOR_SECONDARY = "#E8EEF9"
    COLOR_SECONDARY_TEXT = "#1F2937"
    COLOR_TEXT_MUTED = "#6B7280"
    COLOR_BORDER = "#D6E0F5"

    COLOR_SUCCESS = "#2E8540"
    COLOR_WARNING = "#F59E0B"
    COLOR_DANGER = "#DC2626"

    CONTROL_HEIGHT = 40
    CONTROL_RADIUS = 10
    SECTION_SPACING = 16

    @staticmethod
    def font(size=None, weight=QFont.Normal):
        font = QFont(UITheme.FONT_FAMILY, size or UITheme.FONT_SIZE_BASE)
        font.setWeight(weight)
        return font

    @staticmethod
    def title_font():
        font = QFont(UITheme.FONT_FAMILY, UITheme.FONT_SIZE_TITLE, QFont.Bold)
        return font

    @staticmethod
    def subtitle_font(weight=QFont.Medium):
        font = QFont(UITheme.FONT_FAMILY, UITheme.FONT_SIZE_SUBTITLE, weight)
        return font


def style_primary_button(button: QPushButton):
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(UITheme.CONTROL_HEIGHT)
    button.setFont(UITheme.subtitle_font())
    button.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {UITheme.COLOR_PRIMARY};
            color: #FFFFFF;
            border: none;
            border-radius: {UITheme.CONTROL_RADIUS}px;
            padding: 0 {UITheme.SECTION_SPACING}px;
        }}
        QPushButton:hover {{
            background-color: {UITheme.COLOR_PRIMARY_HOVER};
        }}
        QPushButton:pressed {{
            background-color: {UITheme.COLOR_PRIMARY_PRESSED};
        }}
        QPushButton:disabled {{
            background-color: {UITheme.COLOR_SECONDARY};
            color: {UITheme.COLOR_TEXT_MUTED};
        }}
        """
    )


def style_secondary_button(button: QPushButton):
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(UITheme.CONTROL_HEIGHT)
    button.setFont(UITheme.subtitle_font())
    button.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {UITheme.COLOR_SURFACE};
            color: {UITheme.COLOR_SECONDARY_TEXT};
            border: 1px solid {UITheme.COLOR_BORDER};
            border-radius: {UITheme.CONTROL_RADIUS}px;
            padding: 0 {UITheme.SECTION_SPACING}px;
        }}
        QPushButton:hover {{
            background-color: {UITheme.COLOR_SECONDARY};
        }}
        QPushButton:pressed {{
            background-color: {UITheme.COLOR_BORDER};
        }}
        QPushButton:disabled {{
            color: {UITheme.COLOR_TEXT_MUTED};
            background-color: {UITheme.COLOR_BACKGROUND};
        }}
        """
    )


def style_text_button(button: QPushButton):
    button.setCursor(Qt.PointingHandCursor)
    button.setFlat(True)
    button.setFont(UITheme.font())
    button.setStyleSheet(
        f"""
        QPushButton {{
            background-color: transparent;
            color: {UITheme.COLOR_PRIMARY};
            border: none;
            padding: 4px {UITheme.SECTION_SPACING // 2}px;
            text-decoration: underline;
        }}
        QPushButton:hover {{
            color: {UITheme.COLOR_PRIMARY_HOVER};
        }}
        QPushButton:pressed {{
            color: {UITheme.COLOR_PRIMARY_PRESSED};
        }}
        """
    )


def style_input(widget: QLineEdit):
    widget.setMinimumHeight(UITheme.CONTROL_HEIGHT)
    widget.setFont(UITheme.font())
    widget.setStyleSheet(
        f"""
        QLineEdit {{
            border: 1px solid {UITheme.COLOR_BORDER};
            border-radius: {UITheme.CONTROL_RADIUS}px;
            padding: 0 {UITheme.SECTION_SPACING}px;
            background-color: {UITheme.COLOR_SURFACE};
        }}
        QLineEdit:focus {{
            border: 2px solid {UITheme.COLOR_PRIMARY};
        }}
        QLineEdit:disabled {{
            background-color: {UITheme.COLOR_SECONDARY};
            color: {UITheme.COLOR_TEXT_MUTED};
        }}
        """
    )


def apply_dialog_frame(dialog: QDialog):
    dialog.setStyleSheet(
        f"""
        QDialog {{
            background-color: {UITheme.COLOR_SURFACE};
        }}
        QLabel {{
            font-family: {UITheme.FONT_FAMILY};
            color: {UITheme.COLOR_SECONDARY_TEXT};
        }}
        """
    )


def create_card_frame(parent=None):
    frame = QFrame(parent)
    frame.setObjectName("CardFrame")
    frame.setStyleSheet(
        f"""
        QFrame#CardFrame {{
            background-color: {UITheme.COLOR_SURFACE};
            border-radius: {UITheme.CONTROL_RADIUS}px;
            border: 1px solid {UITheme.COLOR_BORDER};
        }}
        """
    )
    return frame


def create_section_card(title: str, description: str = "", parent=None):
    """Create a standard card with title (and optional description)."""
    frame = create_card_frame(parent)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2)
    layout.setSpacing(UITheme.SECTION_SPACING)

    title_label = QLabel(title)
    title_label.setFont(UITheme.subtitle_font())
    layout.addWidget(title_label)

    if description:
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setFont(UITheme.font())
        desc_label.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
        layout.addWidget(desc_label)

    return frame, layout


def add_form_row(container_layout: QVBoxLayout, label_text: str, widget: QWidget):
    row = QHBoxLayout()
    row.setSpacing(UITheme.SECTION_SPACING // 2)
    label = QLabel(label_text)
    label.setFont(UITheme.font())
    label.setMinimumWidth(120)
    row.addWidget(label)
    row.addWidget(widget)
    container_layout.addLayout(row)
    return row


def add_card_row(container_layout: QVBoxLayout, cards):
    """将多个卡片横向排列后添加到给定的垂直布局中。"""

    row_layout = QHBoxLayout()
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(UITheme.SECTION_SPACING)

    for item in cards:
        widget = item
        stretch = 1
        if isinstance(item, tuple):
            widget, stretch = item

        if isinstance(widget, QWidget):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        row_layout.addWidget(widget, stretch)

    container_layout.addLayout(row_layout)
    return row_layout


def create_header_divider():
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Plain)
    line.setFixedWidth(1)
    line.setStyleSheet(f"background-color: {UITheme.COLOR_BORDER};")
    line.setFixedHeight(UITheme.CONTROL_HEIGHT)
    line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return line


def create_pill_button(text: str, active: bool = False, parent=None):
    button = QPushButton(text, parent)
    button.setCursor(Qt.PointingHandCursor)
    button.setCheckable(True)
    button.setChecked(active)
    button.setMinimumHeight(max(32, UITheme.CONTROL_HEIGHT - 6))
    radius = button.minimumHeight() // 2
    base_border = UITheme.COLOR_PRIMARY if active else UITheme.COLOR_BORDER
    base_color = UITheme.COLOR_PRIMARY if active else UITheme.COLOR_SECONDARY_TEXT
    base_bg = "rgba(37, 99, 235, 0.12)" if active else "transparent"
    button.setStyleSheet(
        f"""
        QPushButton {{
            border-radius: {radius}px;
            border: 1px solid {base_border};
            background-color: {base_bg};
            color: {base_color};
            padding: 0 {UITheme.SECTION_SPACING}px;
        }}
        QPushButton:hover {{
            border-color: {UITheme.COLOR_PRIMARY_HOVER};
            color: {UITheme.COLOR_PRIMARY};
        }}
        QPushButton:checked {{
            border-color: {UITheme.COLOR_PRIMARY};
            background-color: rgba(37, 99, 235, 0.12);
            color: {UITheme.COLOR_PRIMARY};
        }}
        QPushButton:checked:hover {{
            border-color: {UITheme.COLOR_PRIMARY_HOVER};
        }}
        QPushButton:disabled {{
            border-color: {base_border};
            background-color: {base_bg};
            color: {base_color};
        }}
        """
    )
    if active:
        button.setEnabled(False)
    return button


def build_vision_mode_actions(main_window, active_key: str):
    items = [
        ("图片推理", 2, "image"),
        ("视频推理", 3, "video"),
        ("摄像头检测", 4, "camera"),
        ("设置与帮助", 5, "settings"),
    ]
    actions = []
    for label, page_idx, key in items:
        btn = create_pill_button(label, active=(key == active_key))
        if key != active_key:
            btn.clicked.connect(lambda _=False, idx=page_idx: main_window.gotoPage(idx))
        actions.append(btn)
    return actions


def build_settings_mode_actions(main_window, active_key: str):
    items = [
        ("推理参数", 5, "core"),
        ("MQTT 上传", 7, "mqtt"),
        ("WebDAV 上传", 8, "webdav"),
        ("OTA 更新", 9, "ota"),
    ]
    actions = []
    for label, page_idx, key in items:
        btn = create_pill_button(label, active=(key == active_key))
        if key != active_key:
            btn.clicked.connect(lambda _=False, idx=page_idx: main_window.gotoPage(idx))
        actions.append(btn)
    return actions


def create_navigation_card(title: str, description: str, parent=None):
    card = create_card_frame(parent)
    card.setObjectName("NavigationCard")
    card.setCursor(Qt.PointingHandCursor)
    card.setStyleSheet(
        f"""
        QFrame#NavigationCard {{
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: {UITheme.CONTROL_RADIUS * 2}px;
            border: 1px solid {UITheme.COLOR_BORDER};
        }}
        QFrame#NavigationCard:hover {{
            border: 1px solid {UITheme.COLOR_PRIMARY};
            background-color: #FFFFFF;
        }}
        """
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2,
                              UITheme.SECTION_SPACING * 2)
    layout.setSpacing(UITheme.SECTION_SPACING // 2)

    title_label = QLabel(title)
    title_label.setFont(UITheme.subtitle_font(QFont.DemiBold))
    layout.addWidget(title_label)

    desc_label = QLabel(description)
    desc_label.setWordWrap(True)
    desc_label.setFont(UITheme.font())
    desc_label.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
    layout.addWidget(desc_label)

    layout.addStretch()
    return card

###############################################################################
#   登录/注册/修改密码 模块
###############################################################################
class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册")
        self.setFixedSize(400, 300)
        apply_dialog_frame(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2)
        layout.setSpacing(UITheme.SECTION_SPACING)
        title = QLabel("创建新账户")
        title.setFont(UITheme.title_font())
        layout.addWidget(title, alignment=Qt.AlignLeft)
        # 用户名输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        style_input(self.edit_user)
        layout.addWidget(self.edit_user)
        # 密码输入
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("密码")
        self.edit_pass.setEchoMode(QLineEdit.Password)
        style_input(self.edit_pass)
        layout.addWidget(self.edit_pass)
        # 确认密码
        self.edit_pass2 = QLineEdit()
        self.edit_pass2.setPlaceholderText("确认密码")
        self.edit_pass2.setEchoMode(QLineEdit.Password)
        style_input(self.edit_pass2)
        layout.addWidget(self.edit_pass2)
        # 注册按钮
        btn_register = QPushButton("确认注册", clicked=self.do_register)
        style_primary_button(btn_register)
        layout.addWidget(btn_register)

    def do_register(self):
        username = self.edit_user.text().strip()
        pw1 = self.edit_pass.text()
        pw2 = self.edit_pass2.text()
        # 简单校验
        if not username or not pw1:
            QMessageBox.warning(self, "警告", "用户名和密码不能为空！")
            return
        if pw1 != pw2:
            QMessageBox.warning(self, "警告", "两次输入的密码不一致！")
            return
        users = load_users()
        if username in users:
            QMessageBox.warning(self, "警告", "用户名已存在，请换一个。")
            return
        # 保存新用户
        users[username] = hash_password(pw1)
        try:
            save_users(users)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法保存用户数据: {e}")
            return
        QMessageBox.information(self, "成功", "注册成功！请返回登录。")
        self.close()


class ChangePasswordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("修改密码")
        self.setFixedSize(400, 350)
        apply_dialog_frame(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2)
        layout.setSpacing(UITheme.SECTION_SPACING)
        title = QLabel("更新账户密码")
        title.setFont(UITheme.title_font())
        layout.addWidget(title, alignment=Qt.AlignLeft)
        # 用户名输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        style_input(self.edit_user)
        layout.addWidget(self.edit_user)
        # 旧密码输入
        self.edit_old_pass = QLineEdit()
        self.edit_old_pass.setPlaceholderText("当前密码")
        self.edit_old_pass.setEchoMode(QLineEdit.Password)
        style_input(self.edit_old_pass)
        layout.addWidget(self.edit_old_pass)
        # 新密码输入
        self.edit_new_pass = QLineEdit()
        self.edit_new_pass.setPlaceholderText("新密码")
        self.edit_new_pass.setEchoMode(QLineEdit.Password)
        style_input(self.edit_new_pass)
        layout.addWidget(self.edit_new_pass)
        # 确认新密码
        self.edit_new_pass2 = QLineEdit()
        self.edit_new_pass2.setPlaceholderText("确认新密码")
        self.edit_new_pass2.setEchoMode(QLineEdit.Password)
        style_input(self.edit_new_pass2)
        layout.addWidget(self.edit_new_pass2)
        # 确认修改按钮
        btn_change = QPushButton("确认修改", clicked=self.do_change)
        style_primary_button(btn_change)
        layout.addWidget(btn_change)

    def do_change(self):
        username = self.edit_user.text().strip()
        old_pw = self.edit_old_pass.text()
        new_pw1 = self.edit_new_pass.text()
        new_pw2 = self.edit_new_pass2.text()
        if not username or not old_pw or not new_pw1:
            QMessageBox.warning(self, "警告", "所有字段均不能为空！")
            return
        if new_pw1 != new_pw2:
            QMessageBox.warning(self, "警告", "两次输入的新密码不一致！")
            return
        users = load_users()
        if username not in users:
            QMessageBox.warning(self, "警告", "用户名不存在！")
            return
        old_hash = users.get(username, "")
        if hash_password(old_pw) != old_hash:
            QMessageBox.warning(self, "警告", "当前密码错误！")
            return
        # 更新密码
        users[username] = hash_password(new_pw1)
        try:
            save_users(users)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法保存用户数据: {e}")
            return
        QMessageBox.information(self, "成功", "密码修改成功！请使用新密码登录。")
        self.close()


class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("用户登录")
        self.setFixedSize(400, 300)
        apply_dialog_frame(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2,
                                  UITheme.SECTION_SPACING * 3 // 2)
        layout.setSpacing(UITheme.SECTION_SPACING)
        title = QLabel("欢迎回来")
        title.setFont(UITheme.title_font())
        layout.addWidget(title, alignment=Qt.AlignLeft)
        # 用户名和密码输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        style_input(self.edit_user)
        layout.addWidget(self.edit_user)
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("密码")
        self.edit_pass.setEchoMode(QLineEdit.Password)
        style_input(self.edit_pass)
        layout.addWidget(self.edit_pass)
        # 按钮布局
        btn_login = QPushButton("登录", clicked=self.do_login)
        btn_cancel = QPushButton("取消", clicked=self.reject)
        btn_register = QPushButton("注册新用户", clicked=self.open_register)
        btn_change = QPushButton("修改密码", clicked=self.open_change)
        style_primary_button(btn_login)
        style_secondary_button(btn_cancel)
        style_text_button(btn_register)
        style_text_button(btn_change)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(UITheme.SECTION_SPACING)
        actions_row.addWidget(btn_login)
        actions_row.addWidget(btn_cancel)
        actions_row.addStretch(1)

        links_row = QHBoxLayout()
        links_row.setSpacing(UITheme.SECTION_SPACING // 2)
        btn_register.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn_change.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        links_row.addWidget(btn_register)
        links_row.addWidget(btn_change)
        links_row.addStretch(1)

        layout.addLayout(actions_row)
        layout.addLayout(links_row)
        self.username = None

    def do_login(self):
        username = self.edit_user.text().strip()
        password = self.edit_pass.text()
        if not username or not password:
            QMessageBox.warning(self, "警告", "用户名和密码不能为空！")
            return
        users = load_users()
        if username not in users or hash_password(password) != users.get(username):
            QMessageBox.warning(self, "错误", "用户名或密码不正确！")
        else:
            self.username = username
            QMessageBox.information(self, "欢迎", f"登录成功，欢迎 {username}！")
            self.accept()

    def open_register(self):
        dlg = RegisterDialog(self)
        dlg.exec_()

    def open_change(self):
        dlg = ChangePasswordDialog(self)
        dlg.exec_()


###############################################################################
#          多线程(视频推理 + 摄像头检测)
###############################################################################
class VideoProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    frame_signal    = pyqtSignal(QImage)

    def __init__(self, video_path, model, conf_thres, device_option, parent=None):
        super().__init__(parent)
        self.video_path    = video_path
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self._running      = True

        self.id_map  = {}
        self.next_id = 1
        self.detected_objects = {}  # 保存检测到的螺栓ID和状态
        self.first_frame_orig = None
        self.first_frame_ann  = None

        self.frame_records = []   # 聚合后每个螺栓的最佳检测结果
        self.export_frames = []  # [{frame_id, raw_frame, ann_frame, frame_idx, bolt_id}]
        self.best_records = {}

    def run(self):
        # 每次视频推理前强制清空跟踪器状态，保证 ID 从 1 开始
        reset_tracker_state(self.model)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit("")
            return

        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        out_path = "temp_output_video.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid  = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        results_gen = self.model.track(
            source  = self.video_path,
            conf    = self.conf_thres,
            device  = self.device_option,
            tracker = BOTSORT_CONFIG,
            imgsz   = 640,
            stream  = True
        )

        idx_frame = 0
        for result in results_gen:
            if not self._running:
                break
            idx_frame += 1
            frame_bgr = result.orig_img
            if frame_bgr is None:
                continue

            # --- ID二次映射逻辑（和原代码一致）---
            for box in result.boxes:
                raw_id = box.id
                if raw_id is None:
                    continue
                raw_id = int(raw_id.item()) if hasattr(raw_id, "item") else int(raw_id)
                if raw_id not in self.id_map:
                    self.id_map[raw_id] = self.next_id
                    self.next_id += 1
                stable_id = self.id_map[raw_id]
                if box.data.shape[1] >= 8:
                    box.data[0,7] = stable_id
                box.__dict__["id"] = stable_id
                # 记录检测对象 (首次出现时登记)
                if stable_id not in self.detected_objects:
                    cid = int(box.cls[0]) if box.cls is not None else -1
                    cname = result.names.get(cid, str(cid))
                    self.detected_objects[stable_id] = cname

            # --- 数据收集：按 bolt_id 聚合每个螺栓的最佳结果 ---
            frame_name = build_frame_name(idx_frame)
            ann_bgr_for_export = None
            for box in result.boxes:
                stable_id = getattr(box, "id", None)
                if stable_id is None:
                    continue
                cid = int(box.cls[0]) if box.cls is not None else -1
                cname = result.names.get(cid, str(cid))
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                coords = [round(float(x), 2) for x in box.xyxy[0].tolist()]
                candidate = {
                    "图片ID": frame_name["id"],
                    "frame": idx_frame,
                    "bolt_id": stable_id,
                    "status": cname,
                    "conf": conf,
                    "confidence": conf,
                    "x1": coords[0],
                    "y1": coords[1],
                    "x2": coords[2],
                    "y2": coords[3],
                    "raw_path": frame_name["raw_filename"],
                    "det_path": frame_name["det_filename"],
                }
                previous = self.best_records.get(stable_id)
                best = pick_better_bolt_record(previous, candidate)
                if best is candidate:
                    if ann_bgr_for_export is None:
                        ann_bgr_for_export = result.plot(img=frame_bgr.copy())
                    export_record = dict(candidate)
                    export_record.update(
                        {
                            "frame_idx": idx_frame,
                            "raw_frame": frame_bgr.copy(),
                            "ann_frame": ann_bgr_for_export.copy(),
                        }
                    )
                    self.best_records[stable_id] = export_record

            # --- 首帧保存用于报告 ---
            if idx_frame == 1:
                self.first_frame_orig = frame_bgr.copy()
            ann_bgr = result.plot(img=frame_bgr.copy())
            if idx_frame == 1:
                self.first_frame_ann = ann_bgr.copy()

            out_vid.write(ann_bgr)

            # --- 实时UI显示 ---
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            hh, ww, cc = ann_rgb.shape
            qimg = QImage(ann_rgb.data, ww, hh, ww*3, QImage.Format_RGB888)
            self.frame_signal.emit(qimg)

            prog = int(idx_frame / total * 100)
            self.progress_update.emit(prog)

        out_vid.release()
        self.frame_records = aggregated_records_to_rows(self.best_records, mode="video")
        self.export_frames = [
            {
                "图片ID": record.get("图片ID"),
                "frame_idx": record.get("frame_idx"),
                "bolt_id": bolt_id,
                "raw_frame": record.get("raw_frame"),
                "ann_frame": record.get("ann_frame"),
            }
            for bolt_id, record in sorted(self.best_records.items(), key=lambda item: str(item[0]))
            if record.get("raw_frame") is not None and record.get("ann_frame") is not None
        ]
        self.detected_objects = {
            bolt_id: record.get("status", "")
            for bolt_id, record in sorted(self.best_records.items(), key=lambda item: str(item[0]))
        }
        self.finished_signal.emit(os.path.abspath(out_path))

    def stop(self):
        self._running = False
        self.wait()


class CameraCaptureThread(QThread):
    frame_signal    = pyqtSignal(QImage)
    finished_signal = pyqtSignal(dict)
    error_signal    = pyqtSignal(str)

    def __init__(
        self,
        model,
        conf_thres,
        device_option,
        save_root_dir,
        capture_mode="video",
        camera_context=None,
        parent=None,
    ):
        super().__init__(parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self.save_root_dir = save_root_dir
        self.capture_mode  = capture_mode if capture_mode in {"video", "images"} else "video"
        self.camera_context = dict(camera_context or {})

        self.scan_layout = create_scan_layout(self.save_root_dir, "c")
        self.scan_dir = self.scan_layout["scan_dir"]
        base_name     = os.path.basename(self.scan_dir)
        parts         = base_name.split("_")
        if len(parts) >= 3:
            self.scan_timestamp = f"{parts[1]}_{parts[2]}"
        else:
            self.scan_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.video_path   = os.path.join(self.scan_layout["raw_part"], f"Video_{self.scan_timestamp}.mp4")
        self.saved_frames = []
        self.scan_metadata = refresh_scan_id(self.scan_layout, "c", extra_metadata=self.camera_context)
        self._running     = True

    def run(self):
        self.scan_metadata = refresh_scan_id(self.scan_layout, "c", extra_metadata=self.camera_context)
        try:
            camera_info = open_camera_device(self.camera_context.get("camera_index"))
            cap = camera_info["capture"]
            runtime_context = dict(camera_info.get("metadata") or {})
            runtime_context["camera_position"] = self.camera_context.get("camera_position", "")
            self.camera_context.update(runtime_context)
            self.scan_metadata = refresh_scan_id(self.scan_layout, "c", extra_metadata=self.camera_context)
        except Exception as e:
            self.error_signal.emit(f"摄像头打开失败：{e}")
            return

        writer = None
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720),
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 1.0:
            fps = 20.0

        if self.capture_mode == "video":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.video_path, fourcc, fps, frame_size)
            if not writer.isOpened():
                cap.release()
                self.error_signal.emit("视频写入器初始化失败。")
                return

        frame_index = 0
        success = False
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    continue

                if self.capture_mode == "video":
                    writer.write(frame)
                else:
                    frame_index += 1
                    img_path = os.path.join(self.scan_layout["raw_part"], build_image_name(frame_index)["raw_filename"])
                    if cv2.imwrite(img_path, frame):
                        self.saved_frames.append(img_path)

                ann_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hh, ww, _ = ann_rgb.shape
                qimg = QImage(ann_rgb.data, ww, hh, ww * 3, QImage.Format_RGB888)
                self.frame_signal.emit(qimg)
                success = True
        finally:
            self._running = False
            cap.release()
            if writer is not None:
                writer.release()

        info = {
            "success": success,
            "scan_dir": self.scan_dir,
            "scan_layout": dict(self.scan_layout),
            "mode": self.capture_mode,
            "video_path": self.video_path if self.capture_mode == "video" and success else "",
            "frames": list(self.saved_frames),
            "message": "" if success else "摄像头未捕获到有效画面。",
            "camera_context": dict(self.camera_context),
        }
        self.finished_signal.emit(info)

    def stop(self):
        self._running = False
        self.wait()


###############################################################################
#           公共函数 / UI基类
###############################################################################
def pil_to_pixmap(pil_img):
    pil_img = pil_img.convert("RGB")
    data    = pil_img.tobytes("raw", "RGB")
    w, h = pil_img.size
    qimg = QImage(data, w, h, w*3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def create_scan_layout(save_root_dir, suffix):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"scan_{now}_{suffix}"
    scan_dir = os.path.join(save_root_dir, folder)
    image_part = os.path.join(scan_dir, "image_part")
    raw_part = os.path.join(scan_dir, "raw_part")
    text_part = os.path.join(scan_dir, "text_part")
    for dir_path in (scan_dir, image_part, raw_part, text_part):
        os.makedirs(dir_path, exist_ok=True)
    return {
        "scan_dir": scan_dir,
        "image_part": image_part,
        "raw_part": raw_part,
        "text_part": text_part,
    }


def ensure_scan_layout(scan_dir):
    image_part = os.path.join(scan_dir, "image_part")
    raw_part = os.path.join(scan_dir, "raw_part")
    text_part = os.path.join(scan_dir, "text_part")
    for dir_path in (scan_dir, image_part, raw_part, text_part):
        os.makedirs(dir_path, exist_ok=True)
    return {
        "scan_dir": scan_dir,
        "image_part": image_part,
        "raw_part": raw_part,
        "text_part": text_part,
    }




def build_media_name(prefix, index, *, raw_ext=".jpg", det_suffix="_det", det_ext=None):
    raw_ext = raw_ext if raw_ext.startswith(".") else f".{raw_ext}"
    det_ext = det_ext or raw_ext
    det_ext = det_ext if det_ext.startswith(".") else f".{det_ext}"
    media_id = f"{prefix}{index}"
    return {
        "id": media_id,
        "raw_filename": f"{media_id}{raw_ext}",
        "det_filename": f"{media_id}{det_suffix}{det_ext}",
    }


def build_image_name(index, *, raw_ext=".jpg", det_ext=None):
    return build_media_name("IMAGE", index, raw_ext=raw_ext, det_suffix="_DET", det_ext=det_ext)


def build_frame_name(index, *, raw_ext=".jpg", det_ext=None):
    return build_media_name("frame", index, raw_ext=raw_ext, det_suffix="_det", det_ext=det_ext)

def make_scan_dir(save_root_dir, suffix):
    return create_scan_layout(save_root_dir, suffix)["scan_dir"]

def get_all_batches(save_root_dir):
    """
    扫描本地所有检测批次文件夹（scan_YYYYMMDD_HHMMSS_*），
    返回 [{'path': 批次目录, 'type': p/v/c, 'time': 时间字符串, 'name': 文件夹名}, ...]，按时间降序排列。
    """
    batches = []
    for name in os.listdir(save_root_dir):
        if name.startswith("scan_") and os.path.isdir(os.path.join(save_root_dir, name)):
            # 例如 scan_20240708_140102_v
            parts = name.split("_")
            if len(parts) >= 4:
                batch_type = parts[-1]
                batch_time = "_".join(parts[1:-1])
                batches.append({
                    "path": os.path.join(save_root_dir, name),
                    "type": batch_type,
                    "time": batch_time,
                    "name": name
                })
    # 按时间逆序排列
    batches.sort(key=lambda b: b["time"], reverse=True)
    return batches

class FunctionPage(QWidget):
    """
    带“返回上一页”按钮+标题的基类
    """
    def __init__(self, main_window, title_str, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setObjectName("FunctionPage")
        self.setStyleSheet(
            f"""
            QWidget#FunctionPage {{
                background-color: {UITheme.COLOR_BACKGROUND};
                color: {UITheme.COLOR_SECONDARY_TEXT};
            }}
            QLabel {{
                font-family: {UITheme.FONT_FAMILY};
            }}
            """
        )

        base = QVBoxLayout(self)
        base.setContentsMargins(UITheme.SECTION_SPACING * 2,
                                UITheme.SECTION_SPACING * 2,
                                UITheme.SECTION_SPACING * 2,
                                UITheme.SECTION_SPACING * 2)
        base.setSpacing(int(UITheme.SECTION_SPACING * 1.5))

        header = QHBoxLayout()
        header.setSpacing(UITheme.SECTION_SPACING)

        btn_back = QPushButton("返回上一页", clicked=self.on_back)
        style_secondary_button(btn_back)
        btn_back.setFont(UITheme.font())
        btn_back.setMinimumWidth(140)
        header.addWidget(btn_back, alignment=Qt.AlignLeft)

        title_container = QVBoxLayout()
        title_container.setSpacing(UITheme.SECTION_SPACING // 3)

        title_label = QLabel(title_str)
        title_label.setObjectName("PageTitle")
        title_label.setFont(UITheme.title_font())
        title_label.setAlignment(Qt.AlignLeft)
        title_container.addWidget(title_label)

        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("PageSubtitle")
        self.subtitle_label.setFont(UITheme.font())
        self.subtitle_label.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
        self.subtitle_label.hide()
        title_container.addWidget(self.subtitle_label)

        header.addLayout(title_container, stretch=1)

        self._actions_host = QWidget()
        self._actions_host.setObjectName("HeaderActionsHost")
        self._actions_host.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self._actions_layout = QHBoxLayout(self._actions_host)
        self._actions_layout.setContentsMargins(0, 0, 0, 0)
        self._actions_layout.setSpacing(UITheme.SECTION_SPACING // 2)
        header.addWidget(self._actions_host, alignment=Qt.AlignRight)
        self._actions_host.hide()
        base.addLayout(header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        base.addWidget(self.scroll_area)

        self._content_host = QWidget()
        self.scroll_area.setWidget(self._content_host)
        self.content_layout = QVBoxLayout(self._content_host)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(UITheme.SECTION_SPACING)
        self.content_layout.setAlignment(Qt.AlignTop)

    def on_back(self):
        pass

    def set_subtitle(self, text: str):
        if text:
            self.subtitle_label.setText(text)
            self.subtitle_label.show()
        else:
            self.subtitle_label.hide()

    def clear_header_actions(self):
        while self._actions_layout.count():
            item = self._actions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def set_header_actions(self, widgets):
        self.clear_header_actions()
        for widget in widgets:
            self._actions_layout.addWidget(widget)
        if widgets:
            self._actions_host.show()
        else:
            self._actions_host.hide()


class RecentBatchList(QWidget):
    """显示最近检测批次的简易列表，辅助用户快速回溯结果。"""

    def __init__(self, main_window, allowed_types=None, empty_text="暂无历史批次", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.allowed_types = allowed_types
        self.empty_text = empty_text

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(UITheme.SECTION_SPACING // 2)

        self.refresh()

    def refresh(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        batches = get_all_batches(self.main_window.save_root_dir)
        if self.allowed_types:
            batches = [b for b in batches if b.get("type") in self.allowed_types]

        if not batches:
            label = QLabel(self.empty_text)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
            label.setFont(UITheme.font())
            self._layout.addWidget(label)
            self._layout.addStretch()
            return

        for batch in batches[:3]:
            self._layout.addWidget(self._create_entry(batch))

        self._layout.addStretch()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _create_entry(self, batch):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(UITheme.SECTION_SPACING // 2)

        info = QLabel(self._format_batch_text(batch))
        info.setFont(UITheme.font())
        layout.addWidget(info)

        btn_open = QPushButton("打开目录")
        style_text_button(btn_open)
        btn_open.clicked.connect(lambda _=False, path=batch.get("path"): self._open_folder(path))
        layout.addWidget(btn_open)
        layout.addStretch()
        return widget

    def _format_batch_text(self, batch):
        raw_time = batch.get("time", "")
        time_display = raw_time
        try:
            time_display = datetime.datetime.strptime(raw_time, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        type_map = {"p": "图片批次", "v": "视频批次", "c": "摄像头批次"}
        type_text = type_map.get(batch.get("type"), "其他批次")
        name = batch.get("name", "")
        return f"{time_display} · {type_text} · {name}"

    def _open_folder(self, path):
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.warning(self, "提示", "对应目录不存在或已被移动。")


###############################################################################
#   图片推理页面
###############################################################################
class ImageInferencePage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw, "图片推理", parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self.folder_sort_reverse = False
        self.folder_sort_mode = "manual"  # manual | auto
        self.folder_scan_count = 0
        self.set_header_actions(build_vision_mode_actions(self.main_window, "image"))
        self.initUI()

    def initUI(self):
        quick_card, quick_layout = create_section_card(
            "开始检测",
            "选择单张图片或整个文件夹，系统会自动运行推理并生成检测归档。",
        )
        action_row = QHBoxLayout()
        action_row.setSpacing(UITheme.SECTION_SPACING)

        btn_sel = QPushButton("选择图片文件", clicked=self.select_image)
        style_primary_button(btn_sel)
        action_row.addWidget(btn_sel)

        btn_dir = QPushButton("选择图片文件夹", clicked=self.select_folder)
        style_secondary_button(btn_dir)
        action_row.addWidget(btn_dir)

        self.btn_sort_mode = QPushButton()
        style_secondary_button(self.btn_sort_mode)
        self.btn_sort_mode.clicked.connect(self.toggle_sort_mode)
        action_row.addWidget(self.btn_sort_mode)

        self.btn_sort_order = QPushButton()
        style_secondary_button(self.btn_sort_order)
        self.btn_sort_order.clicked.connect(self.toggle_folder_sort)
        self._update_sort_button_label()
        action_row.addWidget(self.btn_sort_order)
        action_row.addStretch()

        self._update_sort_mode_button_label()
        quick_layout.addLayout(action_row)
        self.content_layout.addWidget(quick_card)

        preview_card, preview_layout = create_section_card(
            "检测预览",
            "左侧显示原始图像，右侧显示模型标注结果，可根据窗口大小自适应。",
        )
        image_row = QHBoxLayout()
        image_row.setSpacing(UITheme.SECTION_SPACING)

        self.label_orig = QLabel("等待选择图片")
        self.label_orig.setMinimumSize(360, 240)
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_orig.setStyleSheet(
            f"background-color: {UITheme.COLOR_BACKGROUND};"
            f"border: 2px dashed {UITheme.COLOR_BORDER};"
            f"border-radius: {UITheme.CONTROL_RADIUS}px;"
        )

        self.label_res = QLabel("检测结果将在此展示")
        self.label_res.setMinimumSize(360, 240)
        self.label_res.setAlignment(Qt.AlignCenter)
        self.label_res.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_res.setStyleSheet(
            f"background-color: {UITheme.COLOR_BACKGROUND};"
            f"border: 2px dashed {UITheme.COLOR_BORDER};"
            f"border-radius: {UITheme.CONTROL_RADIUS}px;"
        )

        image_row.addWidget(self.label_orig)
        image_row.addWidget(self.label_res)
        preview_layout.addLayout(image_row)

        detail_card, detail_layout = create_section_card(
            "检测详情",
            "推理后的螺栓识别记录将以结构化文本方式输出。",
        )
        self.text_detail = QTextEdit()
        self.text_detail.setReadOnly(True)
        self.text_detail.setMinimumHeight(180)
        self.text_detail.setStyleSheet(
            f"border: 1px solid {UITheme.COLOR_BORDER};"
            f"border-radius: {UITheme.CONTROL_RADIUS}px;"
        )
        detail_layout.addWidget(self.text_detail)

        add_card_row(
            self.content_layout,
            [
                (preview_card, 3),
                (detail_card, 2),
            ],
        )

        history_card, history_layout = create_section_card(
            "最近图片批次",
            "系统会自动列出最近完成的图片检测批次，方便快速回溯结果归档。",
        )
        self.history_list = RecentBatchList(self.main_window, allowed_types={"p"})
        history_layout.addWidget(self.history_list)
        self.content_layout.addWidget(history_card)

    def select_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not fp:
            return
        scan_layout = create_scan_layout(self.main_window.save_root_dir, "p")
        refresh_scan_id(scan_layout, "p")
        scan_dir = scan_layout["scan_dir"]
        try:
            rows, sample_orig, sample_ann, annotated_infos = self.run_inference([fp], scan_layout)
            self.archive_results(rows, sample_orig, sample_ann, annotated_infos, scan_layout)
            self.history_list.refresh()
            QMessageBox.information(
                self, "检测结果归档完成", f"本次检测所有结果已保存到：\n{scan_dir}"
            )
            mw = self.main_window
            if getattr(mw, "webdav_upload_mode", "manual") == "auto" and getattr(
                mw, "webdav_host", ""
            ):
                try:
                    dav = WebDAVUploader(
                        host=mw.webdav_host,
                        username=mw.webdav_user,
                        password=mw.webdav_pass,
                        remote_path=mw.webdav_remote_path,
                    )
                    dav.upload_batch(scan_dir, resume=True)
                    QMessageBox.information(self, "上传完成", "WebDAV 上传成功")
                except Exception as e:
                    QMessageBox.warning(self, "上传失败", f"WebDAV 上传失败：{e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误: {e}")

    def select_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
        if not dir_path:
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if self.folder_sort_mode == "auto":
            self.folder_sort_reverse = self._next_auto_reverse()
        names = [
            f
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in exts
        ]
        def _num_key(name):
            m = re.search(r"\d+", name)
            return int(m.group()) if m else float("inf")
        files = [
            os.path.join(dir_path, f)
            for f in sorted(names, key=_num_key, reverse=self.folder_sort_reverse)
        ]
        if not files:
            QMessageBox.warning(self, "警告", "该文件夹内未找到图片")
            return
        scan_layout = create_scan_layout(self.main_window.save_root_dir, "p")
        refresh_scan_id(scan_layout, "p")
        scan_dir = scan_layout["scan_dir"]
        progress_dialog = QProgressDialog("正在检测图片...", "", 0, len(files), self)
        progress_dialog.setWindowTitle("检测进度")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.setValue(0)

        def update_progress(done, total):
            progress_dialog.setLabelText(f"正在检测图片（{done}/{total}）")
            progress_dialog.setValue(done)
            QApplication.processEvents()

        progress_dialog.show()
        try:
            self.folder_scan_count += 1
            rows, sample_orig, sample_ann, annotated_infos = self.run_inference(
                files,
                scan_layout,
                progress_callback=update_progress,
                sort_reverse=self.folder_sort_reverse,
            )
            progress_dialog.setValue(len(files))
            self.archive_results(rows, sample_orig, sample_ann, annotated_infos, scan_layout)
            self.history_list.refresh()
            QMessageBox.information(
                self,
                "检测完成",
                f"已处理 {len(files)} 张图片，结果保存在：\n{scan_dir}",
            )
            mw = self.main_window
            if getattr(mw, "webdav_upload_mode", "manual") == "auto" and getattr(
                mw, "webdav_host", "",
            ):
                try:
                    dav = WebDAVUploader(
                        host=mw.webdav_host,
                        username=mw.webdav_user,
                        password=mw.webdav_pass,
                        remote_path=mw.webdav_remote_path,
                    )
                    dav.upload_batch(scan_dir, resume=True)
                    QMessageBox.information(self, "上传完成", "WebDAV 上传成功")
                except Exception as e:
                    QMessageBox.warning(self, "上传失败", f"WebDAV 上传失败：{e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误: {e}")
        finally:
            progress_dialog.close()
            self._update_sort_button_label()

    def toggle_folder_sort(self):
        self.folder_sort_reverse = not self.folder_sort_reverse
        self._update_sort_button_label()

    def _update_sort_button_label(self):
        if self.folder_sort_mode == "auto":
            auto_reverse = self._next_auto_reverse()
            if auto_reverse:
                self.btn_sort_order.setText("自动顺序：下一次倒序")
                self.btn_sort_order.setToolTip("自动模式：奇数批次正序，偶数批次倒序（下一次将倒序）")
            else:
                self.btn_sort_order.setText("自动顺序：下一次正序")
                self.btn_sort_order.setToolTip("自动模式：奇数批次正序，偶数批次倒序（下一次将正序）")
            self.btn_sort_order.setEnabled(False)
        else:
            if self.folder_sort_reverse:
                self.btn_sort_order.setText("读取顺序：名称倒序")
                self.btn_sort_order.setToolTip("按文件名倒序读取文件夹中的图片")
            else:
                self.btn_sort_order.setText("读取顺序：名称升序")
                self.btn_sort_order.setToolTip("按文件名升序读取文件夹中的图片")
            self.btn_sort_order.setEnabled(True)

    def toggle_sort_mode(self):
        self.folder_sort_mode = "auto" if self.folder_sort_mode == "manual" else "manual"
        self._update_sort_mode_button_label()
        self._update_sort_button_label()

    def _update_sort_mode_button_label(self):
        if self.folder_sort_mode == "auto":
            self.btn_sort_mode.setText("顺序选择：自动")
            self.btn_sort_mode.setToolTip("自动轮换排序：奇数批次正序，偶数批次倒序")
        else:
            self.btn_sort_mode.setText("顺序选择：手动")
            self.btn_sort_mode.setToolTip("手动切换文件名顺序/倒序读取")

    def _next_auto_reverse(self):
        # 奇数批次正序，偶数批次倒序；计数从1开始
        return (self.folder_scan_count + 1) % 2 == 0

    def run_inference(self, files, scan_layout, progress_callback=None, sort_reverse=False):
        image_part = scan_layout["image_part"]
        raw_part = scan_layout["raw_part"]
        # 为每次批量扫描创建全新的模型实例，确保跟踪器状态绝对独立
        model_for_batch = YOLO(self.main_window.model_weight_path)

        # 确保每次批量检测前重置跟踪器，避免上一功能的 ID 计数影响本次结果
        reset_tracker_state(model_for_batch)

        files = sorted(
            files,
            key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group())
            if re.search(r"\d+", os.path.basename(p))
            else float("inf"),
            reverse=sort_reverse,
        )
        total = len(files)
        if progress_callback:
            progress_callback(0, total)
        results_gen = model_for_batch.track(
            source=files,
            conf=self.conf_thres,
            device=self.device_option,
            tracker=BOTSORT_CONFIG,
            imgsz=640,
            stream=True,
            persist=True,
            verbose=False,
        )

        best_records = {}
        sample_orig = None
        sample_ann = None
        detail_lines = []
        annotated_infos = []  # 聚合后 [(image_id, annotated_path)]

        for idx, (fp, res) in enumerate(zip(files, results_gen), start=1):
            orig_bgr = res.orig_img
            ann_bgr = res.plot()
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            image_name = build_image_name(idx)
            image_id = image_name["id"]
            raw_path = os.path.join(raw_part, image_name["raw_filename"])
            ann_path = os.path.join(image_part, image_name["det_filename"])

            if idx == 1:
                sample_orig = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
                sample_ann = Image.fromarray(ann_rgb)
                w0, h0 = sample_orig.size
                target_w = max(self.label_orig.width(), self.label_orig.minimumWidth())
                target_h = max(self.label_orig.height(), self.label_orig.minimumHeight())
                r = min(target_w / w0, target_h / h0, 1.0)
                w1, h1 = max(int(w0 * r), 1), max(int(h0 * r), 1)
                self.label_orig.setPixmap(
                    pil_to_pixmap(sample_orig.resize((w1, h1), Image.Resampling.LANCZOS))
                )
                self.label_res.setPixmap(
                    pil_to_pixmap(sample_ann.resize((w1, h1), Image.Resampling.LANCZOS))
                )

            detections = []
            fallback_counter = 1
            for box in res.boxes:
                cid = int(box.cls[0]) if box.cls is not None else -1
                cname = res.names.get(cid, str(cid))
                cconf = float(box.conf[0]) if box.conf is not None else 0.0
                coords = [float(x) for x in box.xyxy[0].tolist()]
                raw_id = getattr(box, "id", None)
                if raw_id is not None:
                    rid = int(raw_id.item()) if hasattr(raw_id, "item") else int(raw_id)
                else:
                    rid = f"{image_id}#{fallback_counter}"
                    fallback_counter += 1
                detections.append(
                    {
                        "box": box,
                        "coords": coords,
                        "class": cname,
                        "conf": cconf,
                        "raw_id": rid,
                    }
                )

            for det in detections:
                det["stable_id"] = det["raw_id"]

            for det in detections:
                x1, y1, x2, y2 = [round(x, 2) for x in det["coords"]]
                candidate = {
                    "图片ID": image_id,
                    "bolt_id": det["stable_id"],
                    "class": det["class"],
                    "status": det["class"],
                    "confidence": det["conf"],
                    "conf": det["conf"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "raw_path": raw_path,
                    "det_path": ann_path,
                    "raw_image": orig_bgr.copy(),
                    "ann_image": ann_bgr.copy(),
                    "source_file": fp,
                }
                best_records[det["stable_id"]] = pick_better_bolt_record(
                    best_records.get(det["stable_id"]), candidate
                )

            if progress_callback:
                progress_callback(idx, total)

        if hasattr(model_for_batch, "tracker") and model_for_batch.tracker is not None:
            try:
                model_for_batch.tracker.tracker.reset()
            except Exception:
                pass

        # 主动释放本次推理专用的模型实例，彻底断开与上一批次的状态关联
        del model_for_batch

        rows = aggregated_records_to_rows(best_records, mode="image")
        detail_lines = []
        for record in best_records.values():
            try:
                ext = os.path.splitext(record.get("source_file") or "")[1].lower()
                src = record.get("source_file")
                raw_path = record.get("raw_path")
                if src and raw_path and os.path.abspath(src) != os.path.abspath(raw_path):
                    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        shutil.copy2(src, raw_path)
                    else:
                        cv2.imwrite(raw_path, record.get("raw_image"))
                elif raw_path and record.get("raw_image") is not None and not os.path.exists(raw_path):
                    cv2.imwrite(raw_path, record.get("raw_image"))
            except Exception:
                pass
            try:
                det_path = record.get("det_path")
                if det_path and record.get("ann_image") is not None:
                    success = cv2.imwrite(det_path, record.get("ann_image"))
                    if success:
                        annotated_infos.append((record.get("图片ID"), det_path))
            except Exception:
                pass
            detail_lines.append(
                f"{record.get('图片ID')} - ID {record.get('bolt_id')} - {record.get('class')}({float(record.get('confidence', 0.0)):.3f})"
            )

        self.text_detail.setPlainText("\n".join(detail_lines))
        return rows, sample_orig, sample_ann, annotated_infos

    def archive_results(self, rows, sample_orig, sample_ann, ann_infos, scan_layout):
        text_part = scan_layout["text_part"]
        rows = sorted(
            rows,
            key=lambda r: (
                str(r.get("图片ID", "")),
                str(r.get("bolt_id", "")),
            ),
        )
        report_df = to_unified_report_df(rows)
        xlsx_path = os.path.join(text_part, "bolt_detection_result.xlsx")
        csv_path = os.path.join(text_part, "bolt_detection_result.csv")
        try:
            report_df.to_excel(xlsx_path, index=False)
            report_df.to_csv(csv_path, index=False)
        except Exception as e:
            QMessageBox.warning(self, "报表保存失败", f"检测详情保存失败: {e}")
        try:
            fmt = "PNG"
            orig_buf = io.BytesIO()
            sample_orig.save(orig_buf, format=fmt)
            orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")
            orig_data_uri = f"data:image/{fmt.lower()};base64,{orig_b64}"
            ann_buf = io.BytesIO()
            sample_ann.save(ann_buf, format=fmt)
            ann_b64 = base64.b64encode(ann_buf.getvalue()).decode("utf-8")
            ann_data_uri = f"data:image/{fmt.lower()};base64,{ann_b64}"
            html = []
            html.append("<html><head><meta charset='utf-8'><title>检测报告</title></head><body>")
            html.append("<h1>图片批次检测报告</h1>")
            html.append(
                f"<p><b>示例原始图像：</b><br><img src='{orig_data_uri}' width='600'></p>"
            )
            html.append(
                f"<p><b>示例标注图像：</b><br><img src='{ann_data_uri}' width='600'></p>"
            )
            html.append("<h2>检测结果明细</h2>")
            html.append(
                "<table border='1' cellspacing='0' cellpadding='4'><tr><th>图片ID</th><th>螺栓ID</th><th>状态</th><th>置信度</th></tr>"
            )

            for r in rows:
                html.append(
                    f"<tr><td>{r['图片ID']}</td><td>{r['bolt_id']}</td><td>{r['class']}</td><td>{r['confidence']:.3f}</td></tr>"
                )
            html.append("</table></body></html>")
            report_path = os.path.join(text_part, "bolt_detection_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
        except Exception as e:
            QMessageBox.warning(self, "警告", f"报告生成失败: {e}")
    def on_back(self):
        self.main_window.gotoPage(1)


###############################################################################
#   视频推理页面
###############################################################################
class VideoInferencePage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw, "视频推理", parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self.thread        = None
        self.out_path      = ""
        self.current_scan_layout = None
        self.set_header_actions(build_vision_mode_actions(self.main_window, "video"))
        self.initUI()

    def initUI(self):
        quick_card, quick_layout = create_section_card(
            "加载视频",
            "导入需要检测的监控视频，进度条将实时展示推理状态。",
        )

        btn_sel = QPushButton("选择视频文件", clicked=self.select_video)
        style_primary_button(btn_sel)
        quick_layout.addWidget(btn_sel, alignment=Qt.AlignLeft)

        self.info = QLabel("尚未选择视频")
        self.info.setWordWrap(True)
        self.info.setFont(UITheme.font())
        quick_layout.addWidget(self.info)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        quick_layout.addWidget(self.bar)

        self.content_layout.addWidget(quick_card)

        preview_card, preview_layout = create_section_card(
            "推理预览",
            "实时查看视频首帧与标注结果，便于核验推理效果。",
        )
        self.label_vid = QLabel("等待推理开始")
        self.label_vid.setMinimumSize(400, 260)
        self.label_vid.setAlignment(Qt.AlignCenter)
        self.label_vid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_vid.setStyleSheet(
            f"background-color: {UITheme.COLOR_BACKGROUND};"
            f"border: 2px dashed {UITheme.COLOR_BORDER};"
            f"border-radius: {UITheme.CONTROL_RADIUS}px;"
        )
        preview_layout.addWidget(self.label_vid)

        self.btn_open = QPushButton("打开处理后视频", clicked=self.open_video)
        style_secondary_button(self.btn_open)
        self.btn_open.setEnabled(False)
        preview_layout.addWidget(self.btn_open, alignment=Qt.AlignLeft)

        history_card, history_layout = create_section_card(
            "最近视频批次",
            "展示最近归档的视频检测任务，快速核对导出结果。",
        )
        self.history_list = RecentBatchList(self.main_window, allowed_types={"v"})
        history_layout.addWidget(self.history_list)
        add_card_row(
            self.content_layout,
            [
                (preview_card, 3),
                (history_card, 2),
            ],
        )

    def select_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not fp:
            return
        self.info.setText(f"已选择视频: {os.path.basename(fp)}，开始推理…")
        self.bar.setValue(0)
        self.current_scan_layout = create_scan_layout(self.main_window.save_root_dir, "v")
        refresh_scan_id(self.current_scan_layout, "v")
        # 启动视频处理线程
        self.thread = VideoProcessingThread(
            fp, self.model, self.conf_thres, self.device_option
        )
        self.thread.progress_update.connect(self.bar.setValue)
        self.thread.finished_signal.connect(self.on_finish)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def update_frame(self, qimg):
        target_w = max(self.label_vid.width(), self.label_vid.minimumWidth())
        target_h = max(self.label_vid.height(), self.label_vid.minimumHeight())
        pm = QPixmap.fromImage(qimg).scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_vid.setPixmap(pm)

    def on_finish(self, path):
        if path and os.path.exists(path):
            self.out_path = path
            self.info.setText(f"视频处理完成: {os.path.basename(path)}")
            self.btn_open.setEnabled(True)
            QMessageBox.information(self, "完成", "视频推理完成！")

            # ========= 1. 生成检测报告(HTML) =========
            base_name, ext = os.path.splitext(path)
            report_path = base_name + "_report.html"
            try:
                html = []
                html.append("<html><head><meta charset='utf-8'><title>检测报告</title></head><body>")
                html.append("<h1>视频检测报告</h1>")
                # 如果有首帧图像，加入报告
                if self.thread and self.thread.first_frame_orig is not None:
                    fmt = "JPEG"
                    orig_rgb = ann_rgb = None
                    try:
                        orig_bgr = self.thread.first_frame_orig
                        ann_bgr = self.thread.first_frame_ann
                        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        orig_rgb = ann_rgb = None
                    if orig_rgb is not None and ann_rgb is not None:
                        orig_buf = io.BytesIO()
                        Image.fromarray(orig_rgb).save(orig_buf, format=fmt)
                        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')
                        orig_data_uri = f"data:image/{fmt.lower()};base64,{orig_b64}"
                        ann_buf = io.BytesIO()
                        Image.fromarray(ann_rgb).save(ann_buf, format=fmt)
                        ann_b64 = base64.b64encode(ann_buf.getvalue()).decode('utf-8')
                        ann_data_uri = f"data:image/{fmt.lower()};base64,{ann_b64}"
                        html.append(f"<p><b>原始视频首帧：</b><br><img src='{orig_data_uri}' width='600'></p>")
                        html.append(f"<p><b>标注结果示例帧：</b><br><img src='{ann_data_uri}' width='600'></p>")
                # 检测结果列表
                html.append("<h2>检测结果</h2>")
                aggregated_rows = getattr(self.thread, "frame_records", []) if self.thread else []
                if not aggregated_rows:
                    html.append("<p>未检测到任何目标。</p>")
                else:
                    html.append("<ul>")
                    for row in sorted(
                        aggregated_rows,
                        key=lambda x: (bolt_id_sort_key(x.get("bolt_id", "")), str(x.get("图片ID", ""))),
                    ):
                        html.append(
                            f"<li>螺栓编号 {row['bolt_id']}: 状态 = {row['status']}（置信度 {float(row['conf']):.3f}，图片ID {row['图片ID']}，帧 {row['frame']}）</li>"
                        )
                    html.append("</ul>")
                    html.append("<table border='1' cellspacing='0' cellpadding='4'><tr><th>图片ID</th><th>帧号</th><th>螺栓ID</th><th>状态</th><th>置信度</th></tr>")
                    for row in aggregated_rows:
                        html.append(
                            f"<tr><td>{row['图片ID']}</td><td>{row['frame']}</td><td>{row['bolt_id']}</td><td>{row['status']}</td><td>{float(row['conf']):.3f}</td></tr>"
                        )
                    html.append("</table>")
                # 视频文件链接
                file_name = os.path.basename(path)
                html.append(f"<p>输出视频文件：<a href='{file_name}'>{file_name}</a></p>")
                html.append("</body></html>")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(html))
                QMessageBox.information(self, "报告已生成", f"检测报告已保存:\n{report_path}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"报告生成失败: {e}")

        # ========= 2. 自动批次归档 =========
            scan_layout = self.current_scan_layout or create_scan_layout(self.main_window.save_root_dir, "v")
            refresh_scan_id(scan_layout, "v")
            scan_dir = scan_layout["scan_dir"]

        # 导出统一报表（每个螺栓仅保留最佳结果）
            if hasattr(self.thread, "frame_records"):
                xlsx_path = os.path.join(scan_layout["text_part"], "bolt_detection_result.xlsx")
                csv_path = os.path.join(scan_layout["text_part"], "bolt_detection_result.csv")
                try:
                    aggregated = aggregate_bolt_records(self.thread.frame_records)
                    video_rows = aggregated_records_to_rows(aggregated, mode="video")
                    video_df = to_unified_report_df(video_rows).sort_values(by=["螺栓ID", "图片ID"], kind="stable")
                    video_df.to_excel(xlsx_path, index=False)
                    video_df.to_csv(csv_path, index=False)
                except Exception:
                    pass

        # 导出松动关键帧图片
            for frame_info in getattr(self.thread, "export_frames", []):
                try:
                    cv2.imwrite(
                        os.path.join(scan_layout["raw_part"], build_frame_name(frame_info["frame_idx"])["raw_filename"]),
                        frame_info["raw_frame"],
                    )
                    cv2.imwrite(
                        os.path.join(scan_layout["image_part"], build_frame_name(frame_info["frame_idx"])["det_filename"]),
                        frame_info["ann_frame"],
                    )
                except Exception:
                    pass

        # 复制视频
            video_dst = os.path.join(scan_layout["raw_part"], "temp_output_video.mp4")
            try:
                shutil.copy(path, video_dst)
            except Exception as e:
                QMessageBox.warning(self, "拷贝视频失败", f"视频文件复制失败: {e}")

        # 复制HTML检测报告
            if os.path.exists(report_path):
                try:
                    shutil.copy(
                        report_path,
                        os.path.join(scan_layout["text_part"], os.path.basename(report_path))
                    )
                except Exception as e:
                    QMessageBox.warning(self, "拷贝报告失败", f"报告复制失败: {e}")

            QMessageBox.information(
                self,
                "检测结果归档完成",
                f"本次检测所有结果已保存到：\n{scan_dir}"
            )
            self.history_list.refresh()

            mw = self.main_window
            upload_msgs = []

            # ---- MQTT 自动上传 ----
            # 条件：开启自动上传且配置了服务器地址
            if getattr(mw, 'upload_mode', 'manual') == 'auto' and getattr(mw, 'mqtt_host', ''):
                uploader = None
                try:
                    uploader = MqttUploader(
                        host=mw.mqtt_host,
                        port=mw.mqtt_port,
                        username=mw.mqtt_user,
                        password=mw.mqtt_pass,
                        topic=mw.mqtt_topic
                    )
                    uploader.connect()
                    uploader.upload_batch(scan_dir)
                    upload_msgs.append("MQTT 上传成功")
                except Exception as e:
                    upload_msgs.append(f"MQTT 上传失败：{e}")
                finally:
                    try:
                        if uploader:
                            uploader.disconnect()
                    except Exception:
                        pass
            else:
                upload_msgs.append("MQTT 未启用自动上传或未配置服务器地址")

# ---- WebDAV 自动上传 ----
# 条件：开启自动上传且配置了服务器地址
            if getattr(mw, 'webdav_upload_mode', 'manual') == 'auto' and getattr(mw, 'webdav_host', ''):
                try:
                    dav = WebDAVUploader(
                        host=mw.webdav_host,
                        username=mw.webdav_user,
                        password=mw.webdav_pass,
                        remote_path=mw.webdav_remote_path
                    )
                    dav.upload_batch(scan_dir, resume=True)
                    upload_msgs.append("WebDAV 上传成功")
                except Exception as e:
                    upload_msgs.append(f"WebDAV 上传失败：{e}")
            else:
                upload_msgs.append("WebDAV 未启用自动上传或未配置服务器地址")

            QMessageBox.information(self, "上传状态", "\n".join(upload_msgs))

        else:
            self.info.setText("视频推理失败或中断。")
            QMessageBox.critical(self, "错误", "视频推理失败。")

    def open_video(self):
        if self.out_path:
            # Windows平台可用os.startfile直接打开，其他平台可使用QDesktopServices
            try:
                os.startfile(self.out_path)
            except Exception as e:
                QMessageBox.information(self, "提示", f"请手动打开视频文件:\n{self.out_path}")

    def on_back(self):
        # 停止线程如果在运行
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        self.main_window.gotoPage(1)


###############################################################################
#   摄像头检测页面
###############################################################################
class CameraPage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw, "摄像头检测", parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self.thread        = None
        self.capture_mode  = "video"  # 可根据需要切换为 "images"
        self.set_header_actions(build_vision_mode_actions(self.main_window, "camera"))
        self.initUI()

    def initUI(self):
        control_card, control_layout = create_section_card(
            "实时检测",
            "连接摄像头以获取现场画面，可随时开始或停止推理。",
        )
        button_row = QHBoxLayout()
        button_row.setSpacing(UITheme.SECTION_SPACING)

        self.btn_start = QPushButton("开始检测", clicked=self.start_camera)
        style_primary_button(self.btn_start)
        button_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止检测", clicked=self.stop_camera)
        style_secondary_button(self.btn_stop)
        self.btn_stop.setEnabled(False)
        button_row.addWidget(self.btn_stop)
        button_row.addStretch()

        control_layout.addLayout(button_row)
        self.content_layout.addWidget(control_card)

        preview_card, preview_layout = create_section_card(
            "摄像头预览",
            "实时推理画面会在此展示，便于确认设备状态。",
        )

        self.label_cam = QLabel("摄像头画面")
        self.label_cam.setAlignment(Qt.AlignCenter)
        self.label_cam.setMinimumSize(400, 260)
        self.label_cam.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_cam.setStyleSheet(
            f"background-color: {UITheme.COLOR_BACKGROUND};"
            f"border: 2px dashed {UITheme.COLOR_BORDER};"
            f"border-radius: {UITheme.CONTROL_RADIUS}px;"
        )
        preview_layout.addWidget(self.label_cam)
        self.camera_status = QLabel("未开始检测。")
        self.camera_status.setWordWrap(True)
        self.camera_status.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
        preview_layout.addWidget(self.camera_status)

        history_card, history_layout = create_section_card(
            "最近摄像头批次",
            "当实时检测结束并归档后，最近的摄像头批次将列在此处。",
        )
        self.history_list = RecentBatchList(self.main_window, allowed_types={"c"})
        history_layout.addWidget(self.history_list)
        add_card_row(
            self.content_layout,
            [
                (preview_card, 3),
                (history_card, 2),
            ],
        )

    def resolve_camera_context(self):
        camera_info = open_camera_device()
        cap = camera_info.get("capture")
        if cap is not None:
            cap.release()

        metadata = dict(camera_info.get("metadata") or {})
        camera_identity = metadata.get("camera_identity", "")
        camera_map = load_camera_position_map()
        camera_position = camera_map.get(camera_identity, "")

        if not camera_position:
            msg = QMessageBox(self)
            msg.setWindowTitle("绑定相机位置")
            msg.setIcon(QMessageBox.Question)
            serial_text = metadata.get("camera_serial") or "未读取到稳定序列号（受限 fallback）"
            msg.setText(
                "检测到未登记的摄像头，请选择其归属位置后继续。\n"
                f"序列号：{serial_text}\n"
                f"厂商/型号：{metadata.get('vendor', 'Generic')} / {metadata.get('model', 'Unknown')}"
            )
            if metadata.get("warning"):
                msg.setInformativeText(metadata["warning"])
            left_btn = msg.addButton("左相机", QMessageBox.AcceptRole)
            right_btn = msg.addButton("右相机", QMessageBox.AcceptRole)
            cancel_btn = msg.addButton("取消", QMessageBox.RejectRole)
            msg.exec_()
            clicked = msg.clickedButton()
            if clicked == cancel_btn:
                return None
            camera_position = "left" if clicked == left_btn else "right"
            camera_map[camera_identity] = camera_position
            save_camera_position_map(camera_map)

        metadata["camera_position"] = camera_position
        return metadata

    def start_camera(self):
        if self.thread:
            self.stop_camera()
        try:
            camera_context = self.resolve_camera_context()
        except Exception as e:
            QMessageBox.critical(self, "摄像头错误", f"摄像头初始化失败：{e}")
            return
        if not camera_context:
            return

        warning_text = camera_context.get("warning", "")
        serial_text = camera_context.get("camera_serial") or "无稳定序列号（fallback）"
        self.camera_status.setText(
            f"当前设备：{camera_context.get('vendor', 'Generic')} / {camera_context.get('model', 'Unknown')} | "
            f"serial={serial_text} | position={camera_context.get('camera_position', '')}"
            + (f"\n提示：{warning_text}" if warning_text else "")
        )

        self.thread = CameraCaptureThread(
            self.model,
            self.conf_thres,
            self.device_option,
            self.main_window.save_root_dir,
            capture_mode=self.capture_mode,
            camera_context=camera_context,
        )
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.finished_signal.connect(self.handle_capture_finished)
        self.thread.error_signal.connect(self.handle_capture_error)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.btn_stop.setEnabled(False)

    def update_frame(self, qimg):
        target_w = max(self.label_cam.width(), self.label_cam.minimumWidth())
        target_h = max(self.label_cam.height(), self.label_cam.minimumHeight())
        pm = QPixmap.fromImage(qimg).scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_cam.setPixmap(pm)

    def handle_capture_error(self, message):
        QMessageBox.critical(self, "摄像头错误", message)
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_status.setText(message)

    def handle_capture_finished(self, info):
        self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if not info.get("success", False):
            if info.get("message"):
                QMessageBox.warning(self, "提示", info["message"])
            return

        try:
            self.run_post_detection(info)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测处理失败: {e}")

    def run_post_detection(self, info):
        scan_layout = info.get("scan_layout") or {}
        scan_dir = scan_layout.get("scan_dir") or info.get("scan_dir")
        if not scan_dir or not os.path.isdir(scan_dir):
            QMessageBox.warning(self, "提示", "未找到有效的扫描目录。")
            return

        if not scan_layout:
            scan_layout = ensure_scan_layout(scan_dir)

        refresh_scan_id(scan_layout, "c", extra_metadata=info.get("camera_context"))
        progress_dialog = None

        if info.get("mode") == "images":
            image_paths = info.get("frames", [])
            if not image_paths:
                QMessageBox.information(self, "提示", "未捕获到任何图片帧。")
                return

            progress_dialog = QProgressDialog(
                "正在推理摄像头图片...", "", 0, len(image_paths), self
            )
            progress_dialog.setWindowTitle("推理进度")
            progress_dialog.setCancelButton(None)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.setValue(0)

            def update_image_progress(done, total):
                if progress_dialog is None or not progress_dialog.isVisible():
                    return
                progress_dialog.setLabelText(f"正在推理摄像头图片（{done}/{total}）")
                progress_dialog.setMaximum(max(total, 1))
                progress_dialog.setValue(done)
                QApplication.processEvents()

            progress_dialog.show()

            image_page = self.main_window.page_image
            try:
                rows, sample_orig, sample_ann, annotated_infos = image_page.run_inference(
                    image_paths, scan_layout, progress_callback=update_image_progress
                )
                progress_dialog.setValue(len(image_paths))
            finally:
                progress_dialog.close()
                progress_dialog = None

            image_page.archive_results(rows, sample_orig, sample_ann, annotated_infos, scan_layout)

            try:
                image_df = to_unified_report_df(rows).sort_values(by=["图片ID", "螺栓ID"], kind="stable")
                image_df.to_excel(
                    os.path.join(scan_layout["text_part"], "bolt_detection_result.xlsx"),
                    index=False,
                )
                image_df.to_csv(
                    os.path.join(scan_layout["text_part"], "bolt_detection_result.csv"),
                    index=False,
                )
            except Exception:
                pass

            QMessageBox.information(
                self,
                "检测完成",
                f"摄像头检测已完成，结果保存在：\n{scan_dir}",
            )

        else:
            video_path = info.get("video_path")
            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(self, "提示", "未生成有效的视频文件。")
                return

            progress_dialog = QProgressDialog("正在推理摄像头视频...", "", 0, 100, self)
            progress_dialog.setWindowTitle("推理进度")
            progress_dialog.setCancelButton(None)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.setValue(0)

            def update_video_progress(val):
                if progress_dialog is None or not progress_dialog.isVisible():
                    return
                progress_dialog.setLabelText(f"正在推理摄像头视频（{val}%）")
                progress_dialog.setValue(val)
                QApplication.processEvents()

            thread = VideoProcessingThread(
                video_path, self.model, self.conf_thres, self.device_option
            )
            thread.progress_update.connect(update_video_progress)

            progress_dialog.show()
            try:
                thread.run()
                progress_dialog.setValue(100)
            finally:
                progress_dialog.close()
                progress_dialog = None

            processed_src = os.path.abspath("temp_output_video.mp4")
            dest_video_path = ""
            if os.path.exists(processed_src):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                dest_video_path = os.path.join(scan_layout["image_part"], f"{base_name}_det.mp4")
                try:
                    if os.path.exists(dest_video_path):
                        os.remove(dest_video_path)
                    shutil.move(processed_src, dest_video_path)
                except Exception:
                    dest_video_path = os.path.join(scan_layout["image_part"], os.path.basename(processed_src))
                    try:
                        if os.path.exists(dest_video_path):
                            os.remove(dest_video_path)
                        shutil.move(processed_src, dest_video_path)
                    except Exception:
                        try:
                            os.remove(processed_src)
                        except Exception:
                            pass

            frame_records = getattr(thread, "frame_records", None)
            try:
                aggregated = aggregate_bolt_records(frame_records or [])
                video_rows = aggregated_records_to_rows(aggregated, mode="video")
                df = to_unified_report_df(video_rows).sort_values(by=["螺栓ID", "图片ID"], kind="stable")
                df.to_excel(os.path.join(scan_layout["text_part"], "bolt_detection_result.xlsx"), index=False)
                df.to_csv(os.path.join(scan_layout["text_part"], "bolt_detection_result.csv"), index=False)
            except Exception:
                pass

            for frame_info in getattr(thread, "export_frames", []):
                try:
                    frame_name = build_frame_name(frame_info["frame_idx"])
                    cv2.imwrite(
                        os.path.join(scan_layout["raw_part"], frame_name["raw_filename"]),
                        frame_info["raw_frame"],
                    )
                    cv2.imwrite(
                        os.path.join(scan_layout["image_part"], frame_name["det_filename"]),
                        frame_info["ann_frame"],
                    )
                except Exception:
                    pass

            report_path = os.path.join(scan_layout["text_part"], "video_detection_report.html")
            try:
                html = []
                html.append("<html><head><meta charset='utf-8'><title>检测报告</title></head><body>")
                html.append("<h1>视频检测报告</h1>")
                if (
                    thread.first_frame_orig is not None
                    and thread.first_frame_ann is not None
                ):
                    fmt = "JPEG"
                    try:
                        orig_rgb = cv2.cvtColor(thread.first_frame_orig, cv2.COLOR_BGR2RGB)
                        ann_rgb = cv2.cvtColor(thread.first_frame_ann, cv2.COLOR_BGR2RGB)
                        orig_img = Image.fromarray(orig_rgb)
                        ann_img = Image.fromarray(ann_rgb)
                        orig_buf = io.BytesIO()
                        ann_buf = io.BytesIO()
                        orig_img.save(orig_buf, format=fmt)
                        ann_img.save(ann_buf, format=fmt)
                        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")
                        ann_b64 = base64.b64encode(ann_buf.getvalue()).decode("utf-8")
                        html.append(
                            f"<p><b>原始示例帧：</b><br><img src='data:image/{fmt.lower()};base64,{orig_b64}' width='600'></p>"
                        )
                        html.append(
                            f"<p><b>标注示例帧：</b><br><img src='data:image/{fmt.lower()};base64,{ann_b64}' width='600'></p>"
                        )
                    except Exception:
                        pass

                html.append("<h2>检测结果</h2>")
                frame_records = getattr(thread, "frame_records", []) or []
                if not frame_records:
                    html.append("<p>未检测到任何目标。</p>")
                else:
                    html.append("<ul>")
                    for row in sorted(
                        frame_records,
                        key=lambda x: (bolt_id_sort_key(x.get("bolt_id", "")), str(x.get("图片ID", ""))),
                    ):
                        html.append(
                            f"<li>螺栓编号 {row['bolt_id']}: 状态 = {row['status']}（置信度 {float(row['conf']):.3f}，图片ID {row['图片ID']}，帧 {row['frame']}）</li>"
                        )
                    html.append("</ul>")
                    html.append("<table border='1' cellspacing='0' cellpadding='4'><tr><th>图片ID</th><th>帧号</th><th>螺栓ID</th><th>状态</th><th>置信度</th></tr>")
                    for row in frame_records:
                        html.append(
                            f"<tr><td>{row['图片ID']}</td><td>{row['frame']}</td><td>{row['bolt_id']}</td><td>{row['status']}</td><td>{float(row['conf']):.3f}</td></tr>"
                        )
                    html.append("</table>")

                if dest_video_path:
                    rel_name = os.path.basename(dest_video_path)
                    html.append(f"<p>输出视频文件：<a href='{rel_name}'>{rel_name}</a></p>")
                html.append("</body></html>")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(html))
            except Exception:
                pass

            QMessageBox.information(
                self,
                "检测完成",
                f"摄像头检测已完成，结果保存在：\n{scan_dir}",
            )

        self.history_list.refresh()

    def on_back(self):
        # 停止摄像头线程
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.main_window.gotoPage(1)


###############################################################################
#   设置页面
###############################################################################
class SettingsPage(FunctionPage):
    def __init__(self, mw, parent=None):
        super().__init__(mw, "设置与帮助", parent)
        actions = build_vision_mode_actions(self.main_window, "settings")
        actions.append(create_header_divider())
        actions.extend(build_settings_mode_actions(self.main_window, "core"))
        self.set_header_actions(actions)
        self.initUI()

    def initUI(self):
        self.set_subtitle("集中管理模型推理参数与云端配置，保持多页面操作一致。")

        shortcuts_card, shortcuts_layout = create_section_card(
            "快捷入口",
            "根据需要跳转到云端上传、WebDAV 与 OTA 等扩展配置。",
        )
        shortcut_row = QHBoxLayout()
        shortcut_row.setSpacing(UITheme.SECTION_SPACING)

        btn_upload = QPushButton("MQTT 云端上传设置", clicked=lambda: self.main_window.gotoPage(7))
        style_secondary_button(btn_upload)
        shortcut_row.addWidget(btn_upload)

        btn_webdav_upload = QPushButton("WebDAV 上传设置", clicked=lambda: self.main_window.gotoPage(8))
        style_secondary_button(btn_webdav_upload)
        shortcut_row.addWidget(btn_webdav_upload)

        btn_ota = QPushButton("OTA 设置", clicked=lambda: self.main_window.gotoPage(9))
        style_secondary_button(btn_ota)
        shortcut_row.addWidget(btn_ota)
        shortcut_row.addStretch()

        shortcuts_layout.addLayout(shortcut_row)
        self.content_layout.addWidget(shortcuts_card)

        model_card, model_layout = create_section_card(
            "模型与推理参数",
            "统一管理推理所需的权重文件、置信度阈值及结果保存目录。",
        )

        self.ed_weights = QLineEdit(self.main_window.model_weight_path)
        self.ed_weights.setReadOnly(True)
        style_input(self.ed_weights)
        btn_sel = QPushButton("选择...", clicked=self.select_weight)
        style_secondary_button(btn_sel)
        weight_row = QHBoxLayout()
        weight_row.setSpacing(UITheme.SECTION_SPACING // 2)
        weight_row.addWidget(self.ed_weights)
        weight_row.addWidget(btn_sel)
        model_layout.addLayout(weight_row)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(UITheme.SECTION_SPACING)
        lb2 = QLabel("置信度阈值：")
        lb2.setFont(UITheme.font())
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(0, 100)
        self.sld_conf.setValue(int(self.main_window.conf_thres * 100))
        self.sld_conf.valueChanged.connect(self.on_conf_change)
        self.lb_val = QLabel(f"{self.main_window.conf_thres:.2f}")
        slider_row.addWidget(lb2)
        slider_row.addWidget(self.sld_conf)
        slider_row.addWidget(self.lb_val)
        model_layout.addLayout(slider_row)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(UITheme.SECTION_SPACING // 2)
        self.ed_dir = QLineEdit(self.main_window.save_root_dir)
        self.ed_dir.setReadOnly(True)
        style_input(self.ed_dir)
        btn_sel_dir = QPushButton("选择保存目录", clicked=self.select_save_dir)
        style_secondary_button(btn_sel_dir)
        dir_row.addWidget(self.ed_dir)
        dir_row.addWidget(btn_sel_dir)
        model_layout.addLayout(dir_row)

        help_txt = (
            "• 调整置信度阈值后，新任务会自动生效；\n"
            "• 更换模型权重后系统将自动重新加载；\n"
            "• 保存目录用于统一归档所有检测批次。"
        )
        lb_help = QLabel(help_txt)
        lb_help.setWordWrap(True)
        lb_help.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
        model_layout.addWidget(lb_help)

        self.content_layout.addWidget(model_card)

    def select_weight(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "Model Files (*.pt)"
        )
        if fp:
            self.ed_weights.setText(fp)
            self.main_window.model_weight_path = fp
            self.main_window.reload_model()

    def on_conf_change(self, val):
        c = val / 100.0
        self.lb_val.setText(f"{c:.2f}")
        self.main_window.update_conf_thres(c)

    def on_back(self):
        self.main_window.gotoPage(1)

    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据保存根目录")
        if dir_path:
            self.ed_dir.setText(dir_path)
            self.main_window.save_root_dir = dir_path


###############################################################################
# 视觉首页(带大背景 + 4卡片)
###############################################################################
class VisionHomePage(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.bg_label)

        self.overlay = QWidget(self.bg_label)
        self.overlay.setObjectName("VisionOverlay")
        self.overlay.setStyleSheet("background-color: rgba(255, 255, 255, 0.58);")
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        hero = QWidget(self.overlay)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(80, 120, 80, 80)
        hero_layout.setSpacing(UITheme.SECTION_SPACING * 2)
        hero_layout.setAlignment(Qt.AlignHCenter)

        title = QLabel("视觉监测功能")
        title.setFont(QFont(UITheme.FONT_FAMILY, 38, QFont.Bold))
        title.setStyleSheet("color:#FFFF0000;")
        title.setAlignment(Qt.AlignCenter)
        hero_layout.addWidget(title)

        subtitle = QLabel("根据不同采集方式启动推理任务，查看检测结果与归档记录。")
        subtitle.setFont(UITheme.subtitle_font())
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255,0,0,0.52);")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(subtitle)

        card_area = QWidget()
        grid = QGridLayout(card_area)
        grid.setSpacing(UITheme.SECTION_SPACING * 2)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        cards = [
            ("图片推理", "单次或批量选择图片，完成检测并输出报告。", 2),
            ("视频推理", "导入视频并跟踪检测进度，自动生成关键帧与结果。", 3),
            ("摄像头检测", "启动实时摄像头并查看现场识别反馈。", 4),
            ("设置与帮助", "调整推理参数与说明文档，保持模型配置一致。", 5),
        ]
        for i, (title_text, desc, page_idx) in enumerate(cards):
            card = create_navigation_card(title_text, desc)
            card.setMinimumSize(260, 170)
            card.mousePressEvent = lambda e, idx=page_idx: self.mw.gotoPage(idx)
            grid.addWidget(card, i // 2, i % 2)

        hero_layout.addWidget(card_area)

        back_button = QPushButton("返回首页", clicked=lambda: self.mw.gotoPage(0))
        style_secondary_button(back_button)
        back_button.setMinimumWidth(160)
        hero_layout.addWidget(back_button, alignment=Qt.AlignCenter)

        overlay_layout.addWidget(hero)
        overlay_layout.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg_label.setFixedSize(self.size())
        self.overlay.setFixedSize(self.size())
        pix = QPixmap(CRANE_IMAGE_PATH)
        if not pix.isNull():
            pix = pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.bg_label.setPixmap(pix)
        else:
            self.bg_label.setText("未找到 crane.jpg")

    def gotoFunc(self, idx):
        if idx == 0:
            self.mw.gotoPage(2)
        elif idx == 1:
            self.mw.gotoPage(3)
        elif idx == 2:
            self.mw.gotoPage(4)
        elif idx == 3:
            self.mw.gotoPage(5)


###############################################################################
#   6) 振动监测页面（优化版）
###############################################################################
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

class VibrationPage(FunctionPage):
    def __init__(self, mw, parent=None):
        super().__init__(mw, "振动监测", parent)
        self.raw_time = None
        self.raw_acc = None
        self.initUI()

    def initUI(self):
        self.set_subtitle("导入振动监测数据，快速生成波形与频谱分析图。")

        action_card, action_layout = create_section_card(
            "数据操作",
            "选择 Excel 数据文件后，可在下方切换查看原始波形或频谱分析。",
        )

        button_row = QHBoxLayout()
        button_row.setSpacing(UITheme.SECTION_SPACING)

        self.btn_load = QPushButton("读取 Excel", clicked=self.load_excel)
        style_primary_button(self.btn_load)
        button_row.addWidget(self.btn_load)

        self.btn_plot = QPushButton("原始波形", clicked=self.plot_raw)
        style_secondary_button(self.btn_plot)
        button_row.addWidget(self.btn_plot)

        self.btn_fft = QPushButton("频谱分析", clicked=self.plot_fft)
        style_secondary_button(self.btn_fft)
        button_row.addWidget(self.btn_fft)
        button_row.addStretch()
        action_layout.addLayout(button_row)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(UITheme.SECTION_SPACING)
        stats_row.addWidget(QLabel("样本数："))
        self.lb_samples = QLabel("--")
        stats_row.addWidget(self.lb_samples)
        stats_row.addWidget(QLabel("最大加速度："))
        self.lb_max = QLabel("--")
        stats_row.addWidget(self.lb_max)
        stats_row.addWidget(QLabel("最小加速度："))
        self.lb_min = QLabel("--")
        stats_row.addWidget(self.lb_min)
        stats_row.addStretch()
        action_layout.addLayout(stats_row)

        self.content_layout.addWidget(action_card)

        chart_card, chart_layout = create_section_card(
            "图表展示",
            "使用工具栏可对图表进行缩放、拖拽，便于定位异常点。",
        )

        self.fig = plt.Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(320)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.toolbar = NavigationToolbar(self.canvas, self)

        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas)

        self.content_layout.addWidget(chart_card)

    def load_excel(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择振动数据 Excel", "", "Excel Files (*.xlsx *.xls)"
        )
        if not fp:
            return
        try:
            df = pd.read_excel(fp)
            self.raw_time = df["Time"].values
            self.raw_acc  = df["acc"].values
            # 计算统计信息
            sample_count = len(self.raw_time)
            max_val = np.max(self.raw_acc) if sample_count > 0 else None
            min_val = np.min(self.raw_acc) if sample_count > 0 else None
            # 更新UI显示
            self.lb_samples.setText(str(sample_count))
            self.lb_max.setText(f"{max_val:.3f}" if max_val is not None else "--")
            self.lb_min.setText(f"{min_val:.3f}" if min_val is not None else "--")
            QMessageBox.information(
                self, "成功", f"已读取 {os.path.basename(fp)}，样本数 = {sample_count}"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取失败: {e}")

    def plot_raw(self):
        if self.raw_time is None or self.raw_acc is None:
            QMessageBox.warning(self, "警告", "请先读取 Excel 数据")
            return
        self.ax.clear()
        self.ax.plot(self.raw_time, self.raw_acc, lw=1)
        self.ax.set_title("原始加速度波形")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Acceleration")
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_fft(self):
        if self.raw_acc is None:
            QMessageBox.warning(self, "警告", "请先读取 Excel 数据")
            return
        N = len(self.raw_acc)
        if N < 2:
            QMessageBox.warning(self, "警告", "样本数太少，无法做频谱分析")
            return
        dt = float(self.raw_time[1] - self.raw_time[0])
        yf = np.abs(fft(self.raw_acc))
        xf = fftfreq(N, dt)
        idx = xf >= 0

        self.ax.clear()
        self.ax.plot(xf[idx], yf[idx], lw=1)
        self.ax.set_title("频谱 (FFT)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Amplitude")
        self.fig.tight_layout()
        self.canvas.draw()

    def on_back(self):
        self.main_window.gotoPage(0)

###############################################################################
#   7) 上传模块
###############################################################################
class MqttUploader:
    def __init__(self, host, port, username=None, password=None, topic="bolt/upload"):
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.topic = topic
        self.client = None

    def connect(self):
        self.client = mqtt.Client()
        if self.username:
            self.client.username_pw_set(self.username, self.password)
        self.client.connect(self.host, self.port, 60)
        self.client.loop_start()

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None

    def upload_file(self, filepath):
        with open(filepath, "rb") as f:
            data = f.read()
        filename = os.path.basename(filepath)
        payload = filename.encode() + b"||" + data
        self.client.publish(self.topic, payload, qos=1)
        print(f"已上传文件：{filename} 到Topic: {self.topic}")

    def upload_batch(self, batch_dir):
        for name in os.listdir(batch_dir):
            fp = os.path.join(batch_dir, name)
            if os.path.isfile(fp):
                self.upload_file(fp)

class UploadSettingsPage(FunctionPage):
    def __init__(self, mw, parent=None):
        super().__init__(mw, "云端上传设置", parent)
        actions = build_vision_mode_actions(self.main_window, "settings")
        actions.append(create_header_divider())
        actions.extend(build_settings_mode_actions(self.main_window, "mqtt"))
        self.set_header_actions(actions)
        self.initUI()

    def initUI(self):
        self.set_subtitle("配置 MQTT 云端服务器并决定检测结果的自动上传策略。")

        form_card, form_layout = create_section_card(
            "服务器连接",
            "填写 MQTT 服务器地址、认证信息及主题，确保上传链路畅通。",
        )

        self.ed_host = QLineEdit()
        self.ed_host.setPlaceholderText("例如：mqtt.example.com")
        style_input(self.ed_host)
        add_form_row(form_layout, "服务器地址", self.ed_host)

        self.ed_port = QLineEdit()
        self.ed_port.setPlaceholderText("默认 1883")
        style_input(self.ed_port)
        add_form_row(form_layout, "端口号", self.ed_port)

        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("用户名 (可选)")
        style_input(self.ed_user)
        add_form_row(form_layout, "用户名", self.ed_user)

        self.ed_pass = QLineEdit()
        self.ed_pass.setPlaceholderText("密码 (可选)")
        self.ed_pass.setEchoMode(QLineEdit.Password)
        style_input(self.ed_pass)
        add_form_row(form_layout, "密码", self.ed_pass)

        self.ed_topic = QLineEdit("bolt/upload")
        style_input(self.ed_topic)
        add_form_row(form_layout, "主题 Topic", self.ed_topic)

        self.content_layout.addWidget(form_card)

        mode_card, mode_layout = create_section_card(
            "上传策略",
            "选择自动或手动上传。自动模式将在推理完成后立即推送数据。",
        )

        self.btn_mode = QPushButton("当前为手动上传，点击切换为自动上传", clicked=self.toggle_mode)
        style_secondary_button(self.btn_mode)
        mode_layout.addWidget(self.btn_mode)

        self.btn_upload = QPushButton("立即上传最新批次", clicked=self.upload_latest_batch)
        style_primary_button(self.btn_upload)
        mode_layout.addWidget(self.btn_upload)

        self.content_layout.addWidget(mode_card)

        self.sync_from_global()

    def sync_from_global(self):
        mw = self.main_window
        self.ed_host.setText(mw.mqtt_host)
        self.ed_port.setText(str(mw.mqtt_port))
        self.ed_user.setText(mw.mqtt_user)
        self.ed_pass.setText(mw.mqtt_pass)
        self.ed_topic.setText(mw.mqtt_topic)
        self.update_btn_mode()

    def sync_to_global(self):
        mw = self.main_window
        mw.mqtt_host = self.ed_host.text().strip()
        mw.mqtt_port = int(self.ed_port.text().strip() or 1883)
        mw.mqtt_user = self.ed_user.text().strip()
        mw.mqtt_pass = self.ed_pass.text().strip()
        mw.mqtt_topic = self.ed_topic.text().strip()
        mw.upload_mode = "auto" if self.btn_mode.text().startswith("当前为自动上传") else "manual"

    def update_btn_mode(self):
        mw = self.main_window
        if mw.upload_mode == "manual":
            self.btn_mode.setText("当前为手动上传，点击切换为自动上传")
            self.btn_upload.setEnabled(True)
        else:
            self.btn_mode.setText("当前为自动上传，点击切换为手动上传")
            self.btn_upload.setEnabled(False)

    def toggle_mode(self):
        mw = self.main_window
        if mw.upload_mode == "manual":
            mw.upload_mode = "auto"
        else:
            mw.upload_mode = "manual"
        self.update_btn_mode()

    def upload_latest_batch(self):
        # 保存当前参数到全局
        self.sync_to_global()

        # 找到最新批次
        root = self.main_window.save_root_dir
        batches = get_all_batches(root)
        if batches:
            latest = batches[0]['path']
            uploader = MqttUploader(
                host=self.main_window.mqtt_host,
                port=self.main_window.mqtt_port,
                username=self.main_window.mqtt_user,
                password=self.main_window.mqtt_pass,
                topic=self.main_window.mqtt_topic
            )
            try:
                uploader.connect()
                uploader.upload_batch(latest)
                uploader.disconnect()
                QMessageBox.information(self, "上传完成", f"已上传：{latest}")
            except Exception as e:
                QMessageBox.warning(self, "上传失败", f"上传失败：{e}")
        else:
            QMessageBox.warning(self, "无批次", "未发现可上传的检测批次。")

    def on_back(self):
        self.main_window.gotoPage(5)

class WebDAVUploader:
    def __init__(self, host, username=None, password=None, remote_path="/bolt_upload/"):
        # host如 https://your.webdav.server:port/
        options = {
            'webdav_hostname': host,
            'webdav_login': username or '',
            'webdav_password': password or ''
        }
        self.client = Client(options)
        self.host = host.rstrip('/')
        self.username = username or ''
        self.session = requests.Session()
        if username or password:
            self.session.auth = (self.username, password or '')
        self.remote_path = remote_path if remote_path.endswith('/') else remote_path + '/'
        self.chunk_size = 10 * 1024 * 1024  # 10MB 默认分块大小
        self._chunk_base_url = self._init_chunk_base_url()

    def _init_chunk_base_url(self):
        """根据Nextcloud规则构造分块上传基础URL"""
        if not self.username:
            return None
        parsed = urlparse(self.host)
        marker = f"/remote.php/dav/files/{self.username}"
        if marker not in parsed.path:
            return None
        chunk_path = parsed.path.replace(
            f"/remote.php/dav/files/{self.username}",
            f"/remote.php/dav/uploads/{self.username}",
            1
        )
        chunk_url = urlunparse(parsed._replace(path=chunk_path))
        return chunk_url.rstrip('/')

    def _full_url(self, remote_fp):
        return urljoin(self.host + '/', remote_fp.lstrip('/'))

    def _chunk_url(self, upload_id, chunk_name=None):
        if not self._chunk_base_url:
            return None
        base = f"{self._chunk_base_url}/{upload_id}"
        if chunk_name is None:
            return base
        return f"{base}/{chunk_name}"

    def _get_remote_size(self, remote_fp):
        url = self._full_url(remote_fp)
        try:
            r = self.session.head(url)
            if r.status_code == 200 and 'Content-Length' in r.headers:
                return int(r.headers['Content-Length'])
        except Exception:
            pass
        return 0

    def _mkcol(self, url):
        try:
            r = self.session.request("MKCOL", url)
            if r.status_code in (200, 201, 204, 405):
                return
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"无法创建远程目录: {exc}")

    def _list_existing_chunks(self, upload_id):
        chunk_folder_url = self._chunk_url(upload_id)
        if not chunk_folder_url:
            return []
        url = chunk_folder_url + "/"
        try:
            resp = self.session.request("PROPFIND", url, headers={"Depth": "1"})
        except requests.RequestException:
            return []
        if resp.status_code != 207:
            return []
        try:
            xml_root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            return []
        ns = {"d": "DAV:"}
        chunk_indices = set()
        for response in xml_root.findall("d:response", ns):
            href = response.find("d:href", ns)
            if href is None or not href.text:
                continue
            path = unquote(href.text)
            name = os.path.basename(path.rstrip('/'))
            if not name or name == upload_id:
                continue
            try:
                chunk_indices.add(int(name))
            except ValueError:
                continue
        return sorted(chunk_indices)

    def _generate_upload_id(self, local_filepath):
        stat = os.stat(local_filepath)
        raw = f"{os.path.abspath(local_filepath)}|{stat.st_size}|{int(stat.st_mtime)}"
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _ensure_remote_dir(self, remote_fp):
        directory = os.path.dirname(remote_fp)
        if not directory or directory == '/':
            return
        segments = [seg for seg in directory.split('/') if seg]
        current = ''
        for seg in segments:
            current += '/' + seg
            url = self._full_url(current + '/')
            self._mkcol(url)

    def upload_file(self, local_filepath, resume=True):
        filename = os.path.basename(local_filepath)
        remote_fp = self.remote_path + filename
        if resume:
            if self._chunk_base_url:
                self._upload_file_chunked(local_filepath, remote_fp)
            else:
                self._upload_file_resumable(local_filepath, remote_fp)
        else:
            self.client.upload_sync(remote_path=remote_fp, local_path=local_filepath)
            print(f"WebDAV已上传: {filename} 到 {remote_fp}")

    def _upload_file_resumable(self, local_filepath, remote_fp):
        filename = os.path.basename(local_filepath)
        local_size = os.path.getsize(local_filepath)
        remote_size = self._get_remote_size(remote_fp)
        if remote_size >= local_size:
            print(f"WebDAV已存在: {filename}, 跳过上传")
            return
        url = self._full_url(remote_fp)
        with open(local_filepath, 'rb') as f:
            if remote_size > 0:
                f.seek(remote_size)
            headers = {
                'Content-Range': f'bytes {remote_size}-{local_size-1}/{local_size}'
            }
            r = self.session.put(url, data=f, headers=headers)
            r.raise_for_status()
        print(f"WebDAV已上传: {filename} ({remote_size}->{local_size}) 到 {remote_fp}")

    def _upload_file_chunked(self, local_filepath, remote_fp):
        if not self._chunk_base_url:
            self._upload_file_resumable(local_filepath, remote_fp)
            return
        filename = os.path.basename(local_filepath)
        local_size = os.path.getsize(local_filepath)
        remote_size = self._get_remote_size(remote_fp)
        if remote_size >= local_size:
            print(f"WebDAV已存在: {filename}, 跳过上传")
            return

        self._ensure_remote_dir(remote_fp)

        upload_id = self._generate_upload_id(local_filepath)
        chunk_folder_url = self._chunk_url(upload_id) + '/'
        parent_url = self._chunk_base_url + '/'
        self._mkcol(parent_url)
        self._mkcol(chunk_folder_url)

        existing_chunks = self._list_existing_chunks(upload_id)
        uploaded_set = set(existing_chunks)
        total_chunks = max(1, math.ceil(local_size / self.chunk_size))

        with open(local_filepath, 'rb') as f:
            for index in range(total_chunks):
                if index in uploaded_set:
                    continue
                f.seek(index * self.chunk_size)
                data = f.read(self.chunk_size)
                if not data:
                    break
                chunk_name = f"{index:016d}"
                chunk_url = self._chunk_url(upload_id, chunk_name)
                if chunk_url is None:
                    raise RuntimeError("未正确初始化分块上传URL")
                headers = {
                    'Content-Type': 'application/octet-stream',
                    'OC-Chunked': '1',
                    'OC-Total-Length': str(local_size),
                    'OC-Chunk-Size': str(len(data)),
                    'OC-Chunk-Offset': str(index * self.chunk_size)
                }
                try:
                    resp = self.session.put(chunk_url, data=data, headers=headers)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    raise RuntimeError(f"上传分块 {chunk_name} 失败: {exc}")
                print(f"WebDAV分块上传: {filename} chunk {index}")

        destination_url = self._full_url(remote_fp)
        move_headers = {
            'Destination': destination_url,
            'Overwrite': 'T'
        }
        try:
            resp = self.session.request("MOVE", chunk_folder_url.rstrip('/'), headers=move_headers)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"合并分块失败: {exc}")

        print(f"WebDAV已分块上传: {filename} ({local_size}) 到 {remote_fp}")

    def upload_batch(self, batch_dir, resume=True):
        # 批次目录下所有文件全部上传
        for name in os.listdir(batch_dir):
            fp = os.path.join(batch_dir, name)
            if os.path.isfile(fp):
                self.upload_file(fp, resume=resume)

class WebDAVUploadSettingsPage(FunctionPage):
    def __init__(self, mw, parent=None):
        super().__init__(mw, "WebDAV云端上传设置", parent)
        actions = build_vision_mode_actions(self.main_window, "settings")
        actions.append(create_header_divider())
        actions.extend(build_settings_mode_actions(self.main_window, "webdav"))
        self.set_header_actions(actions)
        self.initUI()

    def initUI(self):
        self.set_subtitle("通过 WebDAV 将检测批次上传到企业网盘或私有云。")

        form_card, form_layout = create_section_card(
            "服务器连接",
            "填写 WebDAV 地址与账号信息，支持 HTTPS 与分块续传。",
        )

        self.ed_host = QLineEdit()
        self.ed_host.setPlaceholderText("例如：https://nextcloud.example.com")
        style_input(self.ed_host)
        add_form_row(form_layout, "服务器地址", self.ed_host)

        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("用户名 (可选)")
        style_input(self.ed_user)
        add_form_row(form_layout, "用户名", self.ed_user)

        self.ed_pass = QLineEdit()
        self.ed_pass.setPlaceholderText("密码 (可选)")
        self.ed_pass.setEchoMode(QLineEdit.Password)
        style_input(self.ed_pass)
        add_form_row(form_layout, "密码", self.ed_pass)

        self.ed_remotepath = QLineEdit("/bolt_upload/")
        style_input(self.ed_remotepath)
        add_form_row(form_layout, "远程目录", self.ed_remotepath)

        self.content_layout.addWidget(form_card)

        mode_card, mode_layout = create_section_card(
            "上传策略",
            "自动模式会在推理完成后立即同步到 WebDAV。",
        )

        self.btn_mode = QPushButton("当前为手动上传，点击切换为自动上传", clicked=self.toggle_mode)
        style_secondary_button(self.btn_mode)
        mode_layout.addWidget(self.btn_mode)

        self.btn_upload = QPushButton("立即上传最新批次", clicked=self.upload_latest_batch)
        style_primary_button(self.btn_upload)
        mode_layout.addWidget(self.btn_upload)

        self.content_layout.addWidget(mode_card)
        self.sync_from_global()

    def sync_from_global(self):
        mw = self.main_window
        self.ed_host.setText(mw.webdav_host)
        self.ed_user.setText(mw.webdav_user)
        self.ed_pass.setText(mw.webdav_pass)
        self.ed_remotepath.setText(mw.webdav_remote_path)
        self.update_btn_mode()

    def sync_to_global(self):
        mw = self.main_window
        mw.webdav_host = self.ed_host.text().strip()
        mw.webdav_user = self.ed_user.text().strip()
        mw.webdav_pass = self.ed_pass.text().strip()
        mw.webdav_remote_path = self.ed_remotepath.text().strip()
        mw.webdav_upload_mode = "auto" if self.btn_mode.text().startswith("当前为自动上传") else "manual"

    def update_btn_mode(self):
        mw = self.main_window
        if mw.webdav_upload_mode == "manual":
            self.btn_mode.setText("当前为手动上传，点击切换为自动上传")
            self.btn_upload.setEnabled(True)
        else:
            self.btn_mode.setText("当前为自动上传，点击切换为手动上传")
            self.btn_upload.setEnabled(False)

    def toggle_mode(self):
        mw = self.main_window
        if mw.webdav_upload_mode == "manual":
            mw.webdav_upload_mode = "auto"
        else:
            mw.webdav_upload_mode = "manual"
        self.update_btn_mode()

    def upload_latest_batch(self):
        # 保存当前参数到全局
        self.sync_to_global()

        # 找到最新批次
        root = self.main_window.save_root_dir
        batches = get_all_batches(root)
        if batches:
            latest = batches[0]['path']
            uploader = WebDAVUploader(
                host=self.main_window.webdav_host,
                username=self.main_window.webdav_user,
                password=self.main_window.webdav_pass,
                remote_path=self.main_window.webdav_remote_path
            )
            try:
                uploader.upload_batch(latest, resume=True)
                QMessageBox.information(self, "上传完成", f"已上传：{latest}")
            except Exception as e:
                QMessageBox.warning(self, "上传失败", f"上传失败：{e}")
        else:
            QMessageBox.warning(self, "无批次", "未发现可上传的检测批次。")

    def on_back(self):
        self.main_window.gotoPage(5)  # 返回设置页或你想返回的页面

class OTADownloadThread(QThread):
    """下载模型文件并汇报进度"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, url, dest, auth=None, parent=None):
        super().__init__(parent)
        self.url = url
        self.dest = dest
        self.auth = auth
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            with requests.Session() as sess:
                if self.auth:
                    sess.auth = self.auth
                head = sess.head(self.url, allow_redirects=True)
                head.raise_for_status()
                total = int(head.headers.get('Content-Length', 0))
                r = sess.get(self.url, stream=True)
                r.raise_for_status()
                downloaded = 0
                with open(self.dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._cancel:
                            raise Exception('cancelled')
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                self.progress.emit(int(downloaded * 100 / total))
            self.finished.emit(True, 'success')
        except Exception as e:
            self.finished.emit(False, str(e))


class OTASettingsPage(FunctionPage):
    """OTA 更新设置页面"""
    def __init__(self, mw, parent=None):
        super().__init__(mw, "OTA 设置", parent)
        self.thread = None
        self.progress = None
        actions = build_vision_mode_actions(self.main_window, "settings")
        actions.append(create_header_divider())
        actions.extend(build_settings_mode_actions(self.main_window, "ota"))
        self.set_header_actions(actions)
        self.initUI()

    def initUI(self):
        self.set_subtitle("检查并下载最新模型权重，保持推理能力一致。")

        card, layout = create_section_card(
            "OTA 配置",
            "填写 manifest 地址后即可检查更新，系统会自动拉取模型文件。",
        )

        self.ed_manifest = QLineEdit(self.main_window.ota_manifest_url)
        self.ed_manifest.setPlaceholderText("例如：https://example.com/ota/manifest.json")
        style_input(self.ed_manifest)
        layout.addWidget(self.ed_manifest)

        self.status_label = QLabel("尚未检查更新")
        self.status_label.setStyleSheet(f"color: {UITheme.COLOR_TEXT_MUTED};")
        layout.addWidget(self.status_label)

        btn_check = QPushButton("检查更新", clicked=self.check_update)
        style_primary_button(btn_check)
        layout.addWidget(btn_check, alignment=Qt.AlignLeft)

        self.content_layout.addWidget(card)

    def on_back(self):
        self.main_window.gotoPage(5)

    def check_update(self):
        url = self.ed_manifest.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请先填写 OTA manifest URL")
            return
        self.main_window.ota_manifest_url = url
        self.status_label.setText("正在检查更新...")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            manifest = r.json()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"获取manifest失败: {e}")
            self.status_label.setText("检查失败，请确认地址后重试。")
            return

        remote_ver = manifest.get('version', '')
        model_path = manifest.get('model', '')
        if not model_path:
            QMessageBox.warning(self, "错误", "manifest缺少model字段")
            self.status_label.setText("manifest 缺少 model 字段。")
            return
        if remote_ver == self.main_window.model_version:
            QMessageBox.information(self, "提示", "已是最新")
            self.status_label.setText("当前模型已是最新版本。")
            return

        model_url = urljoin(url, model_path)
        tmp_path = self.main_window.model_weight_path + '.tmp'
        self.thread = OTADownloadThread(model_url, tmp_path)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(lambda ok, msg: self.on_download_finished(ok, msg, remote_ver))

        self.progress = QProgressDialog("下载更新中...", "取消", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.canceled.connect(self.thread.cancel)
        self.progress.show()

        self.thread.start()

    def on_progress(self, val):
        if self.progress:
            self.progress.setValue(val)

    def on_download_finished(self, ok, msg, remote_ver):
        if self.progress:
            self.progress.close()
        tmp_path = self.main_window.model_weight_path + '.tmp'
        dest = self.main_window.model_weight_path
        bak = dest + '.bak'
        if ok:
            try:
                if os.path.exists(dest):
                    shutil.copy2(dest, bak)
                os.replace(tmp_path, dest)
                with open(self.main_window.model_version_file, 'w', encoding='utf-8') as f:
                    f.write(remote_ver)
                self.main_window.model_version = remote_ver
                self.main_window.reload_model()
                QMessageBox.information(self, "成功", "已更新")
                self.status_label.setText(f"模型已更新到版本 {remote_ver}")
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if os.path.exists(bak):
                    shutil.copy2(bak, dest)
                QMessageBox.warning(self, "失败", f"更新失败并已回滚: {e}")
                self.status_label.setText("更新失败，已尝试回滚。")
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(bak) and not os.path.exists(dest):
                shutil.copy2(bak, dest)
            QMessageBox.warning(self, "失败", f"更新失败: {msg}")
            self.status_label.setText("下载失败，请稍后重试。")

###############################################################################
#   总首页 & 主窗口
###############################################################################
class MainHomePage(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self.initUI()

    def initUI(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(self.bg_label)

        self.overlay = QWidget(self.bg_label)
        self.overlay.setObjectName("MainHomeOverlay")
        self.overlay.setStyleSheet("background-color: rgba(255, 255, 255, 0.58);")
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        hero = QWidget(self.overlay)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(80, 120, 80, 80)
        hero_layout.setSpacing(UITheme.SECTION_SPACING * 2)
        hero_layout.setAlignment(Qt.AlignHCenter)

        title = QLabel("岸桥轨道螺栓松动监测系统")
        title.setFont(QFont(UITheme.FONT_FAMILY, 44, QFont.Bold))
        title.setStyleSheet("color: #FFFF0000;")
        title.setAlignment(Qt.AlignCenter)
        hero_layout.addWidget(title)

        subtitle = QLabel("统一的视觉与振动监测平台，为运维团队提供高效稳定的安全巡检能力。")
        subtitle.setFont(UITheme.subtitle_font())
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255,0,0,0.82);")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(subtitle)

        card_area = QWidget()
        grid = QGridLayout(card_area)
        grid.setSpacing(UITheme.SECTION_SPACING * 2)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        vision_card = create_navigation_card(
            "视觉监测",
            "查看图片、视频或实时摄像头的检测结果，集中管理推理流程。",
        )
        vision_card.setMinimumSize(280, 180)
        vision_card.mousePressEvent = lambda e: self.mw.gotoPage(1)

        vibration_card = create_navigation_card(
            "振动监测",
            "导入振动数据并快速进行波形与频谱分析。",
        )
        vibration_card.setMinimumSize(280, 180)
        vibration_card.mousePressEvent = lambda e: self.mw.gotoPage(6)

        settings_card = create_navigation_card(
            "系统设置",
            "配置上传、OTA 以及模型推理参数，保持平台统一。",
        )
        settings_card.setMinimumSize(280, 180)
        settings_card.mousePressEvent = lambda e: self.mw.gotoPage(5)

        grid.addWidget(vision_card, 0, 0)
        grid.addWidget(vibration_card, 0, 1)
        grid.addWidget(settings_card, 0, 2)

        hero_layout.addWidget(card_area)
        hero_layout.addStretch()

        overlay_layout.addWidget(hero)
        overlay_layout.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg_label.setFixedSize(self.size())
        self.overlay.setFixedSize(self.size())
        pix = QPixmap(CRANE_IMAGE_PATH)
        if not pix.isNull():
            pix = pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.bg_label.setPixmap(pix)
        else:
            self.bg_label.setText("未找到 crane.jpg")

class MainWindow(QMainWindow):
    def __init__(self, current_user=None):
        super().__init__()
        self.setWindowTitle("岸桥轨道螺栓松动监测系统")
        self.resize(1200,800)
        self.current_user = current_user
        self.save_root_dir = os.path.expanduser("~")

        self.mqtt_host = ""
        self.mqtt_port = 1883
        self.mqtt_user = ""
        self.mqtt_pass = ""
        self.mqtt_topic = "bolt/upload"
        self.upload_mode = "manual"  # "manual" or "auto"

        self.webdav_host = ""
        self.webdav_user = ""
        self.webdav_pass = ""
        self.webdav_remote_path = "/bolt_upload/"
        self.webdav_upload_mode = "manual"  # "manual" or "auto"

        self.ota_manifest_url = ""

        # 资源路径由全局常量管理
        self.crane_image_path  = CRANE_IMAGE_PATH
        self.model_weight_path = WEIGHTS_PATH
        self.conf_thres        = 0.7
        self.device_option     = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model             = None

        self.model_version_file = os.path.splitext(self.model_weight_path)[0] + ".version"
        self.model_version = ""
        if os.path.exists(self.model_version_file):
            try:
                with open(self.model_version_file, 'r', encoding='utf-8') as f:
                    self.model_version = f.read().strip()
            except Exception:
                self.model_version = ""

        self.init_model()
        self.initUI()

    def init_model(self):
        try:
            self.model = YOLO(self.model_weight_path)
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"YOLO模型加载失败: {e}")
            sys.exit(1)

    def reload_model(self):
        try:
            self.model = YOLO(self.model_weight_path)
            # 更新各功能页的模型引用
            self.page_image.model  = self.model
            self.page_video.model  = self.model
            self.page_camera.model = self.model
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"加载YOLO模型失败: {e}")

    def update_conf_thres(self, val):
        self.conf_thres = val
        # 更新各页面的阈值
        self.page_image.conf_thres  = val
        self.page_video.conf_thres  = val
        self.page_camera.conf_thres = val

    def initUI(self):
        self.stacked = QStackedWidget()

        self.page_main    = MainHomePage(self)     # 0
        self.page_vision  = VisionHomePage(self)   # 1
        self.page_image   = ImageInferencePage(self, self.model, self.conf_thres, self.device_option)   # 2
        self.page_video   = VideoInferencePage(self, self.model, self.conf_thres, self.device_option)   # 3
        self.page_camera  = CameraPage(self, self.model, self.conf_thres, self.device_option)           # 4
        self.page_setting = SettingsPage(self)                                                           # 5
        self.page_vib     = VibrationPage(self)                                                         # 6
        self.page_upload = UploadSettingsPage(self)                                                     #7
        self.page_webdav_upload = WebDAVUploadSettingsPage(self)                                         #8
        self.page_ota    = OTASettingsPage(self)  #9
               
        for p in [
            self.page_main, self.page_vision, self.page_image,
            self.page_video, self.page_camera, self.page_setting,
            self.page_vib, self.page_upload, self.page_webdav_upload,
            self.page_ota
        ]:
            self.stacked.addWidget(p)
        
        self.setCentralWidget(self.stacked)
        self.stacked.setCurrentIndex(0)
        
    def gotoPage(self, idx):
        self.stacked.setCurrentIndex(idx)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 启动登录对话框
    login = LoginDialog()
    if login.exec_() == QDialog.Accepted:
        user = login.username
        win = MainWindow(current_user=user)
        # 显示主窗口
        win.show()
        sys.exit(app.exec_())
    else:
        # 登录未成功，直接退出
        sys.exit(0)


if __name__ == "__main__":
    main()
