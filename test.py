#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import warnings
import pandas as pd

# 用于屏蔽 PyQt5 的一些弃用警告，对功能无影响
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QMessageBox, QFileDialog, QTextEdit, QSlider,
    QLineEdit, QGroupBox, QProgressBar, QFrame, QGridLayout, QSizePolicy,
    QGraphicsDropShadowEffect
)

# 在界面上展示图片/视频帧时的最大尺寸（不影响推理分辨率）
IMAGE_MAX_WIDTH  = 600
IMAGE_MAX_HEIGHT = 400

def pil_to_pixmap(pil_img):
    """将 PIL 图像转为 QPixmap 以便在 PyQt 的 QLabel 上显示"""
    pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    w,h = pil_img.size
    qimg= QImage(data, w,h, w*3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

#######################################
# 视频推理线程（多目标跟踪 + 二次ID映射 + 强制640 + 改写 box.id）
#######################################
class VideoProcessingThread(QThread):
    progress_update= pyqtSignal(int)
    finished_signal= pyqtSignal(str)
    frame_signal   = pyqtSignal(QImage)

    def __init__(self, video_path, model, conf_thres, device_option, save_folder, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model = model
        self.conf_thres = conf_thres
        self.device_option = device_option
        self.save_folder = save_folder
        self._running = True
        self.id_map = {}
        self.next_id = 1
        # 结果收集
        self.result_records = []   # 每帧每个螺栓的信息
        self.loose_frames = {}     # {bolt_id: frame保存过没有}

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

        out_path= "temp_output_video.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

        # 使用 track() + BOTSort/ByteTrack 并强制输入 640×640
        results_gen = self.model.track(
            source = self.video_path,
            conf   = self.conf_thres,
            device = self.device_option,
            tracker= "botsort.yaml",   # 或者 "bytetrack.yaml"
            imgsz  = 640,
            stream = True
        )

        idx_frame = 0
        for result in results_gen:
            if not self._running:
                break
            idx_frame += 1
            frame_bgr = result.orig_img
            if frame_bgr is None:
                continue

        # 二次ID映射同前...
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

        # 记录结果：ID、类别名、置信度、帧号
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            self.result_records.append({
                "frame": idx_frame,
                "bolt_id": stable_id,
                "label": "tight" if class_name == "tight" else "loose",
                "conf": conf
            })

        # 保存loose帧图片（只保存每个ID第一次出现的loose）
            if class_name == "loose" and stable_id not in self.loose_frames:
                img_path = os.path.join(
                    self.save_folder, f"loose_bolt_{stable_id}_frame_{idx_frame}.jpg"
                )
                cv2.imwrite(img_path, frame_bgr)
                self.loose_frames[stable_id] = img_path

            # 用 YOLO 默认绘制
            ann_bgr= result.plot(img= frame_bgr.copy())

            out_vid.write(ann_bgr)

            # 转为RGB后发送到界面
            ann_rgb= cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            hh,ww,cc= ann_rgb.shape
            qimg= QImage(ann_rgb.data, ww, hh, ww*3, QImage.Format_RGB888)
            self.frame_signal.emit(qimg)

            prog= int(idx_frame/total*100)
            self.progress_update.emit(prog)

        out_vid.release()
        self.finished_signal.emit(os.path.abspath(out_path))

    def save_csv(self):
        if not self.result_records:
            return
         # 聚合每个螺栓的最终判定（可选更多统计字段）
        summary = {}
        for rec in self.result_records:
            bid = rec["bolt_id"]
            if bid not in summary:
                summary[bid] = {"bolt_id": bid, "tight_count":0, "loose_count":0, "max_conf":0, "first_frame": rec["frame"], "final_label": rec["label"]}
            if rec["label"] == "loose":
                summary[bid]["loose_count"] += 1
            else:
                summary[bid]["tight_count"] += 1
            if rec["conf"] > summary[bid]["max_conf"]:
                summary[bid]["max_conf"] = rec["conf"]
            if rec["frame"] < summary[bid]["first_frame"]:
                summary[bid]["first_frame"] = rec["frame"]
            # 可以根据业务选定最终label：如loose多于tight判为loose
            summary[bid]["final_label"] = "loose" if summary[bid]["loose_count"] > summary[bid]["tight_count"] else "tight"

        df = pd.DataFrame(list(summary.values()))
        out_csv = os.path.join(self.save_folder, "bolt_detection_result.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        self.save_csv()

    def stop(self):
        self._running= False
        self.wait()

#######################################
# 摄像头检测线程（仅普通推理）
#######################################
class CameraCaptureThread(QThread):
    frame_signal= pyqtSignal(QImage)

    def __init__(self, model, conf_thres, device_option, parent=None):
        super().__init__(parent)
        self.model        = model
        self.conf_thres   = conf_thres
        self.device_option= device_option
        self._running     = True

    def run(self):
        cap= cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        while self._running:
            ok, frame= cap.read()
            if not ok: continue
            try:
                res= self.model.predict(
                    source= frame,
                    conf= self.conf_thres,
                    device= self.device_option,
                    imgsz=640,
                    augment=False
                )[0]
                ann_bgr= res.plot(img= frame.copy())
            except:
                ann_bgr= frame

            ann_rgb= cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            hh,ww,cc= ann_rgb.shape
            qimg= QImage(ann_rgb.data, ww, hh, ww*3, QImage.Format_RGB888)
            self.frame_signal.emit(qimg)
        cap.release()

    def stop(self):
        self._running= False
        self.wait()

#######################################
# 基础功能页面
#######################################
class FunctionPage(QWidget):
    def __init__(self, main_window, title_str, parent=None):
        super().__init__(parent)
        self.main_window= main_window
        base= QVBoxLayout(self)

        btn_back= QPushButton("返回首页", clicked=self.back_home)
        btn_back.setFixedWidth(120)
        base.addWidget(btn_back, alignment=Qt.AlignLeft)

        title_label= QLabel(title_str)
        title_label.setFont(QFont("Microsoft YaHei",16,QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        base.addWidget(title_label)

        self.content_layout= QVBoxLayout()
        base.addLayout(self.content_layout)
        base.addStretch()
        self.setLayout(base)

    def back_home(self):
        self.main_window.gotoPage(0)

#######################################
# 图片推理页面
#######################################
class ImageInferencePage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw,"图片推理",parent)
        self.model= model
        self.conf_thres= conf_thres
        self.device_option= device_option
        self.initUI()

    def initUI(self):
        btn_sel= QPushButton("选择图片文件", clicked=self.select_image)
        btn_sel.setFixedWidth(200)
        self.content_layout.addWidget(btn_sel, alignment=Qt.AlignCenter)

        self.label_orig= QLabel()
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setFixedSize(IMAGE_MAX_WIDTH+40, IMAGE_MAX_HEIGHT+40)
        self.label_orig.setStyleSheet("border:3px dashed gray;")

        self.label_res= QLabel()
        self.label_res.setAlignment(Qt.AlignCenter)
        self.label_res.setFixedSize(IMAGE_MAX_WIDTH+40, IMAGE_MAX_HEIGHT+40)
        self.label_res.setStyleSheet("border:3px dashed gray;")

        hb= QHBoxLayout()
        hb.addWidget(self.label_orig)
        hb.addWidget(self.label_res)
        self.content_layout.addLayout(hb)

        self.text_detail= QTextEdit()
        self.text_detail.setReadOnly(True)
        self.content_layout.addWidget(self.text_detail)

    def select_image(self):
        fp,_= QFileDialog.getOpenFileName(self,"选择图片","","Images (*.jpg *.jpeg *.png)")
        if not fp: return
        try:
            pil= Image.open(fp).convert("RGB")
            w0,h0= pil.size
            r= min(IMAGE_MAX_WIDTH/w0, IMAGE_MAX_HEIGHT/h0,1.0)
            w1,h1= int(w0*r), int(h0*r)

            # 显示原图
            pil_resize= pil.resize((w1,h1), Image.Resampling.LANCZOS)
            self.label_orig.setPixmap(pil_to_pixmap(pil_resize))

            # 推理
            bgr= cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            res= self.model.predict(source=bgr, conf=self.conf_thres,
                                    device=self.device_option, imgsz=640)[0]
            ann_bgr= res.plot(img=bgr.copy())
            ann_rgb= cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

            ann_pil= Image.fromarray(ann_rgb).resize((w1,h1), Image.Resampling.LANCZOS)
            self.label_res.setPixmap(pil_to_pixmap(ann_pil))

            boxes= res.boxes
            if len(boxes)==0:
                self.text_detail.setPlainText("未检测到任何目标")
            else:
                info=[]
                for i, box in enumerate(boxes):
                    cid= int(box.cls[0])
                    cnam= res.names.get(cid,str(cid))
                    cconf= float(box.conf[0])
                    coords= [round(x,2) for x in box.xyxy[0].tolist()]
                    info.append(f"目标{i+1}: 类别={cnam}, 置信度={cconf:.2f}, 坐标={coords}")
                self.text_detail.setPlainText("\n".join(info))
        except Exception as e:
            QMessageBox.critical(self,"错误",f"发生错误: {e}")

#######################################
# 视频推理页面
#######################################
class VideoInferencePage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw,"视频推理",parent)
        self.model= model
        self.conf_thres= conf_thres
        self.device_option= device_option
        self.thread=None
        self.out_path=""
        self.initUI()

    def initUI(self):
        self.save_folder = os.getcwd()  # 默认当前目录
        btn_save_folder = QPushButton("选择结果保存文件夹", clicked=self.select_save_folder)
        btn_save_folder.setFixedWidth(200)
        self.content_layout.addWidget(btn_save_folder, alignment=Qt.AlignCenter)

        btn_sel= QPushButton("选择视频文件", clicked=self.select_video)
        btn_sel.setFixedWidth(200)
        self.content_layout.addWidget(btn_sel, alignment=Qt.AlignCenter)

        self.info= QLabel("这里显示视频推理进度和结果。", alignment=Qt.AlignCenter)
        self.content_layout.addWidget(self.info)

        self.bar= QProgressBar(); self.bar.setFixedWidth(400)
        self.bar.setValue(0)
        self.content_layout.addWidget(self.bar, alignment=Qt.AlignCenter)

        self.label_vid= QLabel("视频推理画面", alignment=Qt.AlignCenter)
        self.label_vid.setFixedSize(600,400)
        self.label_vid.setStyleSheet("border:3px dashed gray;")
        self.content_layout.addWidget(self.label_vid, alignment=Qt.AlignCenter)

        self.btn_open= QPushButton("打开处理后视频", clicked=self.open_video)
        self.btn_open.setFixedWidth(200)
        self.btn_open.setEnabled(False)
        self.content_layout.addWidget(self.btn_open, alignment=Qt.AlignCenter)

    def select_video(self):
        fp,_= QFileDialog.getOpenFileName(self,"选择视频","","Videos (*.mp4 *.avi *.mov *.mkv)")
        if not fp: return
        self.info.setText(f"已选择视频：{os.path.basename(fp)}，开始推理…")
        self.bar.setValue(0)

        self.thread= VideoProcessingThread(
        fp, self.model, self.conf_thres, self.device_option, self.save_folder
        )
        self.thread.progress_update.connect(self.bar.setValue)
        self.thread.finished_signal.connect(self.on_finish)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def on_finish(self, path):
        if path and os.path.exists(path):
            self.out_path= path
            self.info.setText(f"视频处理完成：{path}")
            self.btn_open.setEnabled(True)
            QMessageBox.information(self,"完成","视频推理完成！")
        else:
            self.info.setText("视频推理失败或中断。")
            QMessageBox.critical(self,"错误","视频推理失败。")

    def update_frame(self, qimg):
        pm= QPixmap.fromImage(qimg).scaled(600,400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_vid.setPixmap(pm)

    def open_video(self):
        if self.out_path:
            os.startfile(self.out_path)

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存结果文件夹", "")
        if folder:
            self.save_folder = folder
        else:
            self.save_folder = os.getcwd()  # 默认用当前目录

#######################################
# 摄像头检测页面
#######################################
class CameraPage(FunctionPage):
    def __init__(self, mw, model, conf_thres, device_option, parent=None):
        super().__init__(mw,"摄像头检测",parent)
        self.model       = model
        self.conf_thres  = conf_thres
        self.device_option= device_option
        self.thread=None
        self.initUI()

    def initUI(self):
        hb= QHBoxLayout()
        self.btn_start= QPushButton("开始检测", clicked=self.start_camera)
        self.btn_start.setFixedWidth(120)
        self.btn_stop= QPushButton("停止检测", clicked=self.stop_camera)
        self.btn_stop.setFixedWidth(120)
        self.btn_stop.setEnabled(False)
        hb.addWidget(self.btn_start); hb.addWidget(self.btn_stop)
        self.content_layout.addLayout(hb)

        self.label_cam= QLabel("摄像头画面", alignment=Qt.AlignCenter)
        self.label_cam.setFixedSize(600,400)
        self.label_cam.setFrameShape(QFrame.Box)
        self.content_layout.addWidget(self.label_cam, alignment=Qt.AlignCenter)

    def start_camera(self):
        # 每次点击“开始检测”时，都会用最新的 self.conf_thres 去生成线程
        if self.thread: self.stop_camera()
        self.thread= CameraCaptureThread(
            self.model, self.conf_thres, self.device_option
        )
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread= None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_frame(self, qimg):
        pm= QPixmap.fromImage(qimg).scaled(600,400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_cam.setPixmap(pm)

#######################################
# 设置与帮助页面
#######################################
class SettingsPage(FunctionPage):
    def __init__(self, mw, parent=None):
        super().__init__(mw,"设置及帮助", parent)
        self.initUI()

    def initUI(self):
        group= QGroupBox("模型与推理设置")
        vb= QVBoxLayout(group)

        # 模型权重
        h1= QHBoxLayout()
        lb= QLabel("模型权重：")
        self.ed_weights= QLineEdit(self.main_window.model_weight_path if self.main_window else "")
        self.ed_weights.setReadOnly(True)
        btn_sel= QPushButton("选择...", clicked=self.select_weight)
        h1.addWidget(lb); h1.addWidget(self.ed_weights); h1.addWidget(btn_sel)
        vb.addLayout(h1)

        # 置信度
        h2= QHBoxLayout()
        lb2= QLabel("置信度阈值：")
        self.sld_conf= QSlider(Qt.Horizontal)
        self.sld_conf.setRange(0,100)
        self.sld_conf.setValue(int(self.main_window.conf_thres*100))
        self.sld_conf.valueChanged.connect(self.on_conf_change)
        self.lb_val= QLabel(f"{self.main_window.conf_thres:.2f}")
        h2.addWidget(lb2); h2.addWidget(self.sld_conf); h2.addWidget(self.lb_val)
        vb.addLayout(h2)

        self.content_layout.addWidget(group)

        help_txt=(
            "使用说明：\n"
            "1. 当你在此处调整置信度阈值后，“图片推理”、“视频推理”和“摄像头检测”三者都将使用新的阈值。\n"
            "2. 若已在摄像头检测页面开始检测，可停止后再开始来生效新的阈值；视频推理同理。\n"
            "3. 通过二次映射ID减少ID跳号，并在摄像头检测里同样使用设置中的置信度阈值。"
        )
        lb_help= QLabel(help_txt)
        lb_help.setWordWrap(True)
        self.content_layout.addWidget(lb_help)

    def select_weight(self):
        fp,_= QFileDialog.getOpenFileName(self,"选择模型权重","","Model Files (*.pt)")
        if fp:
            self.ed_weights.setText(fp)
            if self.main_window:
                self.main_window.model_weight_path= fp
                self.main_window.reload_model()

    def on_conf_change(self, val):
        c= val/100.0
        self.lb_val.setText(f"{c:.2f}")
        # 主窗口的 update_conf_thres 会同时更新图片推理、视频推理、摄像头检测的 conf
        if self.main_window:
            self.main_window.update_conf_thres(c)

#######################################
# 首页页面
#######################################
class HomePage(QWidget):
    def __init__(self, mw, parent=None):
        super().__init__(parent)
        self.mw= mw
        self.initUI()

    def initUI(self):
        layout= QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self.bg_label= QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        self.bg_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        layout.addWidget(self.bg_label)

        self.overlay= QWidget(self.bg_label)
        self.overlay.setStyleSheet("background-color: transparent;")
        vbox= QVBoxLayout(self.overlay)
        vbox.setAlignment(Qt.AlignCenter)
        vbox.setSpacing(30)

        title= QLabel("岸桥轨道螺栓松动监测系统")
        title.setFont(QFont("Microsoft YaHei",48,QFont.Bold))
        title.setStyleSheet("color:#ffffff;")
        title.setAlignment(Qt.AlignCenter)
        vbox.addWidget(title)

        card_area= QWidget()
        grid= QGridLayout(card_area)
        grid.setSpacing(40)
        grid.setContentsMargins(60,30,60,50)

        card_style= """
            QFrame {
                background: rgba(0,0,0,0.6);
                border: none;
                border-radius:15px;
            }
            QFrame:hover {
                background: rgba(0,0,0,0.8);
            }
        """

        btn_names=["图片推理","视频推理","摄像头检测","设置与帮助"]
        for i,name in enumerate(btn_names):
            card= QFrame()
            card.setFixedSize(220,130)
            card.setStyleSheet(card_style)
            shadow= QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(15)
            shadow.setOffset(0,0)
            card.setGraphicsEffect(shadow)

            lab= QLabel(name, card)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("border:none; font-size:22px; font-weight:bold; color:#ffffff;")
            vb= QVBoxLayout(card)
            vb.addWidget(lab, alignment=Qt.AlignCenter)

            card.mousePressEvent= lambda e, idx=i+1: self.goto_function(idx)
            grid.addWidget(card, i//2, i%2, alignment=Qt.AlignCenter)

        vbox.addWidget(card_area, alignment=Qt.AlignCenter)
        layout.addWidget(self.bg_label)
        self.setLayout(layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg_label.setFixedSize(self.size())
        self.overlay.setFixedSize(self.size())
        pix= QPixmap(self.mw.crane_image_path)
        if not pix.isNull():
            pix= pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.bg_label.setPixmap(pix)
        else:
            self.bg_label.setText("未找到岸桥照片")

    def goto_function(self, idx):
        if self.mw:
            self.mw.gotoPage(idx)

#######################################
# 主窗口
#######################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("岸桥轨道螺栓松动监测系统")
        self.resize(1200,800)

        # ★ 修改为你自己的best.pt和背景图
        self.model_weight_path= r"D:\money_project\bolt_loose_detection\best.pt"
        self.crane_image_path = r"D:\money_project\bolt_loose_detection\background.jpg"

        self.conf_thres= 0.25  # 初始置信度
        self.device_option= "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model= None

        self.init_model()
        self.initUI()

    def init_model(self):
        try:
            self.model= YOLO(self.model_weight_path)
        except Exception as e:
            QMessageBox.critical(self,"模型加载失败", f"加载模型失败: {e}")
            sys.exit(1)

    def reload_model(self):
        """当用户在设置中选了新的权重文件后，重新加载模型"""
        try:
            self.model= YOLO(self.model_weight_path)
            self.page_image.model= self.model
            self.page_video.model= self.model
            self.page_camera.model= self.model
        except Exception as e:
            QMessageBox.critical(self,"模型加载失败", f"加载模型失败: {e}")

    def update_conf_thres(self, val):
        """当设置中拖动滑条改变置信度阈值后，更新到所有推理页面"""
        self.conf_thres= val
        self.page_image.conf_thres= val
        self.page_video.conf_thres= val
        self.page_camera.conf_thres= val

    def initUI(self):
        self.stacked= QStackedWidget()
        self.page_home= HomePage(self)
        self.page_image= ImageInferencePage(self,self.model,self.conf_thres,self.device_option)
        self.page_video= VideoInferencePage(self,self.model,self.conf_thres,self.device_option)
        self.page_camera= CameraPage(self,self.model,self.conf_thres,self.device_option)
        self.page_setting= SettingsPage(self)

        self.stacked.addWidget(self.page_home)     # index=0
        self.stacked.addWidget(self.page_image)    # index=1
        self.stacked.addWidget(self.page_video)    # index=2
        self.stacked.addWidget(self.page_camera)   # index=3
        self.stacked.addWidget(self.page_setting)  # index=4

        self.setCentralWidget(self.stacked)

    def gotoPage(self, idx):
        self.stacked.setCurrentIndex(idx)

def main():
    app= QApplication(sys.argv)
    app.setStyle("Fusion")
    win= MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
