#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import warnings
import hashlib
import base64
import io
import datetime
import cv2
from PIL import Image
import shutil
import pandas as pd
import paho.mqtt.client as mqtt

# —— 新增：动态获取应用根目录 ——
if getattr(sys, "frozen", False):
    # PyInstaller 打包后，资源临时解压到 _MEIPASS
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 用户数据文件路径
USER_DATA_FILE = os.path.join(BASE_DIR, "users.json")

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

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit,
    QSlider, QLineEdit, QGroupBox, QProgressBar, QFrame, QGridLayout,
    QStackedWidget, QDialog, QSizePolicy
)

import cv2
from ultralytics import YOLO
from PIL import Image

###############################################################################
#   登录/注册/修改密码 模块
###############################################################################
class RegisterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册")
        self.setFixedSize(300, 200)
        layout = QVBoxLayout(self)
        # 用户名输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        layout.addWidget(self.edit_user)
        # 密码输入
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("密码")
        self.edit_pass.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_pass)
        # 确认密码
        self.edit_pass2 = QLineEdit()
        self.edit_pass2.setPlaceholderText("确认密码")
        self.edit_pass2.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_pass2)
        # 注册按钮
        btn_register = QPushButton("确认注册", clicked=self.do_register)
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
        self.setFixedSize(300, 200)
        layout = QVBoxLayout(self)
        # 用户名输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        layout.addWidget(self.edit_user)
        # 旧密码输入
        self.edit_old_pass = QLineEdit()
        self.edit_old_pass.setPlaceholderText("当前密码")
        self.edit_old_pass.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_old_pass)
        # 新密码输入
        self.edit_new_pass = QLineEdit()
        self.edit_new_pass.setPlaceholderText("新密码")
        self.edit_new_pass.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_new_pass)
        # 确认新密码
        self.edit_new_pass2 = QLineEdit()
        self.edit_new_pass2.setPlaceholderText("确认新密码")
        self.edit_new_pass2.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_new_pass2)
        # 确认修改按钮
        btn_change = QPushButton("确认修改", clicked=self.do_change)
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
        self.setFixedSize(300, 180)
        layout = QVBoxLayout(self)
        # 用户名和密码输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("用户名")
        layout.addWidget(self.edit_user)
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("密码")
        self.edit_pass.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.edit_pass)
        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_login = QPushButton("登录", clicked=self.do_login)
        btn_register = QPushButton("注册", clicked=self.open_register)
        btn_change = QPushButton("修改密码", clicked=self.open_change)
        btn_cancel = QPushButton("取消", clicked=self.reject)
        btn_layout.addWidget(btn_login)
        btn_layout.addWidget(btn_register)
        btn_layout.addWidget(btn_change)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
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

        self.frame_records = []   # 记录每帧所有螺栓的检测结果
        self.loose_frames = []   # [(frame_image, bolt_id, frame_idx)]

    def run(self):
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

            # --- 数据收集：每帧每个螺栓 ---
            for box in result.boxes:
                # 只收集有映射的
                stable_id = getattr(box, "id", None)
                if stable_id is None:
                    continue
                cid = int(box.cls[0]) if box.cls is not None else -1
                cname = result.names.get(cid, str(cid))
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # 收集所有检测数据（frame, bolt_id, status, conf）
                self.frame_records.append({
                    "frame": idx_frame,
                    "bolt_id": stable_id,
                    "status": cname,
                    "conf": conf
                })

                # 如果是"松动"螺栓，则收集关键帧（确保仅保存每个螺栓的首帧或全部帧，可按需改动）
                if cname.lower() == "loose":  # 注意"loose"应与你模型类别一致
                    ann_bgr = result.plot(img=frame_bgr.copy())
                    self.loose_frames.append(
                        (ann_bgr.copy(), stable_id, idx_frame)
                    )

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
        self.finished_signal.emit(os.path.abspath(out_path))

    def stop(self):
        self._running = False
        self.wait()


class CameraCaptureThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, model, conf_thres, device_option, parent=None):
        super().__init__(parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self._running      = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                res = self.model.predict(
                    source = frame,
                    conf   = self.conf_thres,
                    device = self.device_option,
                    imgsz  = 640
                )[0]
                ann_bgr = res.plot(img=frame.copy())
            except Exception as e:
                ann_bgr = frame

            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            hh, ww, cc = ann_rgb.shape
            qimg = QImage(ann_rgb.data, ww, hh, ww*3, QImage.Format_RGB888)
            self.frame_signal.emit(qimg)

        cap.release()

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

def make_scan_dir(save_root_dir, suffix):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"scan_{now}_{suffix}"
    scan_dir = os.path.join(save_root_dir, folder)
    os.makedirs(scan_dir, exist_ok=True)
    return scan_dir

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

        base = QVBoxLayout(self)
        btn_back = QPushButton("返回上一页", clicked=self.on_back)
        btn_back.setFixedWidth(120)
        btn_back.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )
        base.addWidget(btn_back, alignment=Qt.AlignLeft)

        title_label = QLabel(title_str)
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        base.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        base.addLayout(self.content_layout)
        base.addStretch()
        self.setLayout(base)

    def on_back(self):
        pass


###############################################################################
#   图片推理页面
###############################################################################
class ImageInferencePage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw, "图片推理", parent)
        self.model         = model
        self.conf_thres    = conf_thres
        self.device_option = device_option
        self.initUI()

    def initUI(self):
        btn_sel = QPushButton("选择图片文件", clicked=self.select_image)
        btn_sel.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )
        btn_sel.setFixedWidth(200)
        self.content_layout.addWidget(btn_sel, alignment=Qt.AlignCenter)

        self.label_orig = QLabel()
        self.label_orig.setFixedSize(600, 400)
        self.label_orig.setStyleSheet("border:3px dashed gray;")
        self.label_orig.setAlignment(Qt.AlignCenter)

        self.label_res = QLabel()
        self.label_res.setFixedSize(600, 400)
        self.label_res.setStyleSheet("border:3px dashed gray;")
        self.label_res.setAlignment(Qt.AlignCenter)

        hb = QHBoxLayout()
        hb.addWidget(self.label_orig)
        hb.addWidget(self.label_res)
        self.content_layout.addLayout(hb)

        self.text_detail = QTextEdit()
        self.text_detail.setReadOnly(True)
        self.content_layout.addWidget(self.text_detail)

    def select_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not fp:
            return
        try:
            pil_img = Image.open(fp).convert("RGB")
            w0, h0 = pil_img.size
            # 将原始图像显示
            r = min(600 / w0, 400 / h0, 1.0)
            w1, h1 = int(w0 * r), int(h0 * r)
            pil_resize = pil_img.resize((w1, h1), Image.Resampling.LANCZOS)
            self.label_orig.setPixmap(pil_to_pixmap(pil_resize))

            # 模型推理
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            res = self.model.predict(
                source=bgr,
                conf=self.conf_thres,
                device=self.device_option,
                imgsz=640
            )[0]
            ann_bgr = res.plot(img=bgr.copy())

            # 显示结果图像
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            ann_pil = Image.fromarray(ann_rgb).resize((w1, h1), Image.Resampling.LANCZOS)
            self.label_res.setPixmap(pil_to_pixmap(ann_pil))

            # 显示检测详情
            boxes = res.boxes
            if len(boxes) == 0:
                self.text_detail.setPlainText("未检测到任何目标")
            else:
                info_lines = []
                for i, box in enumerate(boxes):
                    cid   = int(box.cls[0])
                    cnam  = res.names.get(cid, str(cid))
                    cconf = float(box.conf[0]) if box.conf is not None else 0.0
                    coords= [round(x,2) for x in box.xyxy[0].tolist()]
                    info_lines.append(
                        f"目标{i+1}: 类别={cnam}, 置信度={cconf:.3f}, 坐标={coords}"
                    )
                self.text_detail.setPlainText("\n".join(info_lines))
            # 自动生成检测报告 (HTML 文件)
            try:
                base_name, ext = os.path.splitext(fp)
                report_path = base_name + "_report.html"
                # 准备嵌入报告的图像数据（使用 base64 编码）
                # 原始图像
                fmt = "PNG"
                if ext.lower() in [".jpg", ".jpeg"]:
                    fmt = "JPEG"
                orig_buf = io.BytesIO()
                pil_img.save(orig_buf, format=fmt)
                orig_b64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')
                orig_data_uri = f"data:image/{fmt.lower()};base64,{orig_b64}"
                # 标注后图像
                ann_buf = io.BytesIO()
                # 用相同格式保存标注结果
                Image.fromarray(ann_rgb).save(ann_buf, format=fmt)
                ann_b64 = base64.b64encode(ann_buf.getvalue()).decode('utf-8')
                ann_data_uri = f"data:image/{fmt.lower()};base64,{ann_b64}"
                # 组装HTML内容
                html = []
                html.append("<html><head><meta charset='utf-8'><title>检测报告</title></head><body>")
                html.append("<h1>图片检测报告</h1>")
                html.append(f"<p><b>原始图像：</b><br><img src='{orig_data_uri}' width='600'></p>")
                html.append(f"<p><b>标注结果图像：</b><br><img src='{ann_data_uri}' width='600'></p>")
                html.append("<h2>检测结果</h2>")
                if len(boxes) == 0:
                    html.append("<p>未检测到任何目标。</p>")
                else:
                    html.append("<ul>")
                    for i, box in enumerate(boxes):
                        cid = int(box.cls[0])
                        cname = res.names.get(cid, str(cid))
                        # 将目标编号作为螺栓ID
                        bolt_id = i + 1
                        html.append(f"<li>螺栓编号 {bolt_id}: 状态 = {cname}</li>")
                    html.append("</ul>")
                html.append("</body></html>")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(html))
                QMessageBox.information(self, "报告已生成", f"检测报告已保存:\n{report_path}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"报告生成失败: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生错误: {e}")

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
        self.initUI()

    def initUI(self):
        btn_sel = QPushButton("选择视频文件", clicked=self.select_video)
        btn_sel.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )
        btn_sel.setFixedWidth(200)
        self.content_layout.addWidget(btn_sel, alignment=Qt.AlignCenter)

        self.info = QLabel("视频推理进度信息", alignment=Qt.AlignCenter)
        self.content_layout.addWidget(self.info)

        self.bar = QProgressBar()
        self.bar.setFixedWidth(400)
        self.bar.setValue(0)
        self.content_layout.addWidget(self.bar, alignment=Qt.AlignCenter)

        self.label_vid = QLabel("视频画面", alignment=Qt.AlignCenter)
        self.label_vid.setFixedSize(600, 400)
        self.label_vid.setStyleSheet("border:3px dashed gray;")
        self.content_layout.addWidget(self.label_vid, alignment=Qt.AlignCenter)

        self.btn_open = QPushButton("打开处理后视频", clicked=self.open_video)
        self.btn_open.setFixedWidth(200)
        self.btn_open.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )
        self.btn_open.setEnabled(False)
        self.content_layout.addWidget(self.btn_open, alignment=Qt.AlignCenter)

    def select_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not fp:
            return
        self.info.setText(f"已选择视频: {os.path.basename(fp)}，开始推理…")
        self.bar.setValue(0)
        # 启动视频处理线程
        self.thread = VideoProcessingThread(
            fp, self.model, self.conf_thres, self.device_option
        )
        self.thread.progress_update.connect(self.bar.setValue)
        self.thread.finished_signal.connect(self.on_finish)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def update_frame(self, qimg):
        pm = QPixmap.fromImage(qimg).scaled(
            600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
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
                if not self.thread or len(self.thread.detected_objects) == 0:
                    html.append("<p>未检测到任何目标。</p>")
                else:
                    html.append("<ul>")
                    for sid, cname in sorted(self.thread.detected_objects.items()):
                        html.append(f"<li>螺栓编号 {sid}: 状态 = {cname}</li>")
                    html.append("</ul>")
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
            scan_dir = make_scan_dir(self.main_window.save_root_dir, "v")

        # 导出CSV（所有检测数据）
            if hasattr(self.thread, "frame_records"):
                csv_path = os.path.join(scan_dir, "bolt_detection_result.csv")
                pd.DataFrame(self.thread.frame_records).to_csv(csv_path, index=False)

        # 导出松动关键帧图片
            loose_frames = getattr(self.thread, "loose_frames", [])
            for img, bolt_id, frame_idx in loose_frames:
                img_path = os.path.join(
                    scan_dir, f"loose_bolt_{bolt_id}_frame_{frame_idx}.jpg"
                )
                cv2.imwrite(img_path, img)

        # 复制视频
            video_dst = os.path.join(scan_dir, "temp_output_video.mp4")
            try:
                shutil.copy(path, video_dst)
            except Exception as e:
                QMessageBox.warning(self, "拷贝视频失败", f"视频文件复制失败: {e}")

        # 复制HTML检测报告
            if os.path.exists(report_path):
                try:
                    shutil.copy(
                        report_path,
                        os.path.join(scan_dir, os.path.basename(report_path))
                    )
                except Exception as e:
                    QMessageBox.warning(self, "拷贝报告失败", f"报告复制失败: {e}")

            QMessageBox.information(
                self,
                "检测结果归档完成",
                f"本次检测所有结果已保存到：\n{scan_dir}"
            )

            mw = self.main_window
            if mw.upload_mode == "auto":
                uploader = MqttUploader(
                    host=mw.mqtt_host,
                    port=mw.mqtt_port,
                    username=mw.mqtt_user,
                    password=mw.mqtt_pass,
                    topic=mw.mqtt_topic
                )
            try:
                uploader.connect()
                uploader.upload_batch(scan_dir)
                uploader.disconnect()
                QMessageBox.information(self, "自动上传完成", f"批次数据已上传至云端服务器。")
            except Exception as e:
                QMessageBox.warning(self, "自动上传失败", f"上传失败：{e}")

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
        self.initUI()

    def initUI(self):
        hb = QHBoxLayout()
        btn_font = (
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )

        self.btn_start = QPushButton("开始检测", clicked=self.start_camera)
        self.btn_start.setStyleSheet(btn_font)
        self.btn_start.setFixedWidth(120)

        self.btn_stop = QPushButton("停止检测", clicked=self.stop_camera)
        self.btn_stop.setStyleSheet(btn_font)
        self.btn_stop.setFixedWidth(120)
        self.btn_stop.setEnabled(False)

        hb.addWidget(self.btn_start)
        hb.addWidget(self.btn_stop)
        self.content_layout.addLayout(hb)

        self.label_cam = QLabel("摄像头画面", alignment=Qt.AlignCenter)
        self.label_cam.setFixedSize(600, 400)
        self.label_cam.setStyleSheet("border:3px dashed gray;")
        self.content_layout.addWidget(self.label_cam, alignment=Qt.AlignCenter)

    def start_camera(self):
        if self.thread:
            self.stop_camera()
        self.thread = CameraCaptureThread(
            self.model, self.conf_thres, self.device_option
        )
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_frame(self, qimg):
        pm = QPixmap.fromImage(qimg).scaled(
            600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_cam.setPixmap(pm)

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
        self.initUI()

    def initUI(self):
        group = QGroupBox("模型与推理设置")
        vb = QVBoxLayout(group)

        btn_upload = QPushButton("云端上传设置", clicked=lambda: self.main_window.gotoPage(7))
        self.content_layout.addWidget(btn_upload)  # 或合适的布局位置

        # 模型权重路径
        h1 = QHBoxLayout()
        lb = QLabel("模型权重：")
        self.ed_weights = QLineEdit(self.main_window.model_weight_path)
        self.ed_weights.setReadOnly(True)
        btn_sel = QPushButton("选择...", clicked=self.select_weight)
        btn_sel.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:1px solid white;"
            "border-radius:3px;"
        )
        h1.addWidget(lb)
        h1.addWidget(self.ed_weights)
        h1.addWidget(btn_sel)
        vb.addLayout(h1)

        # 置信度阈值
        h2 = QHBoxLayout()
        lb2 = QLabel("置信度阈值：")
        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(0, 100)
        self.sld_conf.setValue(int(self.main_window.conf_thres * 100))
        self.sld_conf.valueChanged.connect(self.on_conf_change)
        self.lb_val = QLabel(f"{self.main_window.conf_thres:.2f}")
        h2.addWidget(lb2)
        h2.addWidget(self.sld_conf)
        h2.addWidget(self.lb_val)
        vb.addLayout(h2)

        # 添加保存目录选择
        h3 = QHBoxLayout()
        lb3 = QLabel("保存目录：")
        self.ed_dir = QLineEdit(self.main_window.save_root_dir)
        self.ed_dir.setReadOnly(True)
        btn_sel_dir = QPushButton("选择...", clicked=self.select_save_dir)
        h3.addWidget(lb3)
        h3.addWidget(self.ed_dir)
        h3.addWidget(btn_sel_dir)
        vb.addLayout(h3)

        self.content_layout.addWidget(group)

        help_txt = (
            "使用说明：\n"
            "1. 此处调整置信度阈值后，图片/视频/摄像头检测都用新阈值；\n"
            "2. 若检测进行中，请停止后再开始才能生效；\n"
            "3. 可切换YOLO权重文件，系统会自动加载。"
        )
        lb_help = QLabel(help_txt)
        lb_help.setWordWrap(True)
        self.content_layout.addWidget(lb_help)

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
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.bg_label)

        self.overlay = QWidget(self.bg_label)
        self.overlay.setStyleSheet("background-color: rgba(0,0,0,0);")
        vbox = QVBoxLayout(self.overlay)
        vbox.setAlignment(Qt.AlignCenter)
        vbox.setSpacing(30)

        title = QLabel("视觉监测功能")
        title.setFont(QFont("Microsoft YaHei", 36, QFont.Bold))
        title.setStyleSheet("color:#ffffff;")
        title.setAlignment(Qt.AlignCenter)
        vbox.addWidget(title)

        card_area = QWidget()
        grid = QGridLayout(card_area)
        grid.setSpacing(60)
        grid.setContentsMargins(60,30,60,50)

        card_style = """
            QFrame {
                background: rgba(0,0,0,0.6);
                border: none;
                border-radius:15px;
            }
            QFrame:hover {
                background: rgba(0,0,0,0.8);
            }
        """
        names = ["图片推理", "视频推理", "摄像头检测", "设置与帮助"]
        for i, name in enumerate(names):
            card = QFrame()
            card.setFixedSize(300,160)
            card.setStyleSheet(card_style)
            lab = QLabel(name, card)
            lab.setStyleSheet(
                "border:none;"
                "font-size:28px;"
                "font-weight:bold;"
                "color:#ffffff;"
            )
            lab.setAlignment(Qt.AlignCenter)
            vb2 = QVBoxLayout(card)
            vb2.addWidget(lab, alignment=Qt.AlignCenter)
            card.mousePressEvent = lambda e, idx=i: self.gotoFunc(idx)
            grid.addWidget(card, i//2, i%2, alignment=Qt.AlignCenter)

        vbox.addWidget(card_area, alignment=Qt.AlignCenter)
        vbox.addStretch()

        btn_back = QPushButton("返回首页")
        btn_back.setStyleSheet(
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
        )
        btn_back.setFixedWidth(120)
        btn_back.clicked.connect(lambda: self.mw.gotoPage(0))
        vbox.addWidget(btn_back, alignment=Qt.AlignLeft|Qt.AlignBottom)

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

class VibrationPage(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self.raw_time = None
        self.raw_acc  = None

        layout = QVBoxLayout(self)
        btn_font = (
            "color:white;"
            "background-color:black;"
            "border:2px solid white;"
            "border-radius:5px;"
            "font-size:16px;"
            "padding:5px;"
        )

        hb = QHBoxLayout()
        self.btn_load = QPushButton("读取 Excel", clicked=self.load_excel)
        self.btn_load.setStyleSheet(btn_font)
        self.btn_plot = QPushButton("原始波形", clicked=self.plot_raw)
        self.btn_plot.setStyleSheet(btn_font)
        self.btn_fft = QPushButton("频谱分析", clicked=self.plot_fft)
        self.btn_fft.setStyleSheet(btn_font)
        # 平滑功能移除，不添加 self.btn_smooth
        hb.addWidget(self.btn_load)
        hb.addWidget(self.btn_plot)
        hb.addWidget(self.btn_fft)
        layout.addLayout(hb)

        # 图表区域 (Matplotlib 嵌入)
        self.fig = plt.Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1,1,1)
        # 添加导航工具栏以支持缩放平移等交互
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # 数据统计信息
        info_hb = QHBoxLayout()
        info_hb.addWidget(QLabel("样本数:"))
        self.lb_samples = QLabel("--")
        info_hb.addWidget(self.lb_samples)
        info_hb.addWidget(QLabel("最大加速度:"))
        self.lb_max = QLabel("--")
        info_hb.addWidget(self.lb_max)
        info_hb.addWidget(QLabel("最小加速度:"))
        self.lb_min = QLabel("--")
        info_hb.addWidget(self.lb_min)
        layout.addLayout(info_hb)

        btn_back = QPushButton("返回首页", clicked=lambda: self.mw.gotoPage(0))
        btn_back.setStyleSheet(btn_font)
        btn_back.setFixedWidth(120)
        layout.addWidget(btn_back, alignment=Qt.AlignLeft)

        self.setLayout(layout)

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
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # MQTT参数输入区
        self.ed_host = QLineEdit()
        self.ed_host.setPlaceholderText("MQTT服务器地址")
        self.ed_port = QLineEdit()
        self.ed_port.setPlaceholderText("端口号(默认1883)")
        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("用户名(可选)")
        self.ed_pass = QLineEdit()
        self.ed_pass.setPlaceholderText("密码(可选)")
        self.ed_pass.setEchoMode(QLineEdit.Password)
        self.ed_topic = QLineEdit()
        self.ed_topic.setPlaceholderText("上传主题(Topic)")
        self.ed_topic.setText("bolt/upload")

        layout.addWidget(QLabel("服务器地址:"))
        layout.addWidget(self.ed_host)
        layout.addWidget(QLabel("端口号:"))
        layout.addWidget(self.ed_port)
        layout.addWidget(QLabel("用户名:"))
        layout.addWidget(self.ed_user)
        layout.addWidget(QLabel("密码:"))
        layout.addWidget(self.ed_pass)
        layout.addWidget(QLabel("主题(Topic):"))
        layout.addWidget(self.ed_topic)

        # 上传模式切换
        self.btn_mode = QPushButton("当前为手动上传，点击切换为自动上传", clicked=self.toggle_mode)
        layout.addWidget(self.btn_mode)

        # 立即上传按钮（仅手动模式可用）
        self.btn_upload = QPushButton("立即上传最新批次", clicked=self.upload_latest_batch)
        layout.addWidget(self.btn_upload)

        self.content_layout.addLayout(layout)

        # 让UI与全局参数保持同步
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

###############################################################################
#   总首页 & 主窗口
###############################################################################
class MainHomePage(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw = mw
        self.initUI()

    def initUI(self):
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)
        self.layout().setSpacing(0)

        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.bg_label)

        self.overlay = QWidget(self.bg_label)
        self.overlay.setStyleSheet("background-color: rgba(0,0,0,0);")
        vbox = QVBoxLayout(self.overlay)
        vbox.setAlignment(Qt.AlignCenter)
        vbox.setSpacing(30)

        title = QLabel("岸桥轨道螺栓松动监测系统")
        title.setFont(QFont("Microsoft YaHei", 48, QFont.Bold))
        title.setStyleSheet("color:#ffffff;")
        title.setAlignment(Qt.AlignCenter)
        vbox.addWidget(title)

        card_area = QWidget()
        grid = QGridLayout(card_area)
        grid.setSpacing(60)
        grid.setContentsMargins(60,30,60,50)

        card_style = """
            QFrame {
                background: rgba(0,0,0,0.6);
                border: none;
                border-radius:15px;
            }
            QFrame:hover {
                background: rgba(0,0,0,0.8);
            }
        """
        card_vis = QFrame()
        card_vis.setFixedSize(300,160)
        card_vis.setStyleSheet(card_style)
        lab_vis = QLabel("视觉监测", card_vis)
        lab_vis.setStyleSheet("border:none; font-size:28px; font-weight:bold; color:#ffffff;")
        lab_vis.setAlignment(Qt.AlignCenter)
        lay_vis = QVBoxLayout(card_vis)
        lay_vis.addWidget(lab_vis, alignment=Qt.AlignCenter)
        card_vis.mousePressEvent = lambda e: self.mw.gotoPage(1)

        card_vib = QFrame()
        card_vib.setFixedSize(300,160)
        card_vib.setStyleSheet(card_style)
        lab_vib = QLabel("振动监测", card_vib)
        lab_vib.setStyleSheet("border:none; font-size:28px; font-weight:bold; color:#ffffff;")
        lab_vib.setAlignment(Qt.AlignCenter)
        lay_vib = QVBoxLayout(card_vib)
        lay_vib.addWidget(lab_vib, alignment=Qt.AlignCenter)
        card_vib.mousePressEvent = lambda e: self.mw.gotoPage(6)

        grid.addWidget(card_vis, 0,0, alignment=Qt.AlignCenter)
        grid.addWidget(card_vib, 0,1, alignment=Qt.AlignCenter)

        vbox.addWidget(card_area, alignment=Qt.AlignCenter)

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

        # 资源路径由全局常量管理
        self.crane_image_path  = CRANE_IMAGE_PATH
        self.model_weight_path = WEIGHTS_PATH
        self.conf_thres        = 0.7
        self.device_option     = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model             = None

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
        self.page_upload = UploadSettingsPage(self)
               
        for p in [
            self.page_main, self.page_vision, self.page_image,
            self.page_video, self.page_camera, self.page_setting,
            self.page_vib, self.page_upload, 
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