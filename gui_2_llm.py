# -*- coding: utf-8 -*-
"""
Unified GUI:
Top  : Posture Trends — PyQt5 + QtCharts (minutes-based, smooth updates)
Bottom: Live OpenPose/RealSense panel embedded in the same window

- Independent subsystems: the video/OpenPose worker runs in its own QThread.
- UDP receiver prints to terminal + updates "Current Sitting Posture" box + tray + taskbar overlay.
- Excel/CSV auto-attach + watching.
- Windows taskbar overlay dot; size/offset adjustable (see CONTROL HERE).

If pyrealsense2 or pyopenpose are missing:
- RealSense missing → uses webcam.
- OpenPose missing  → shows raw camera frames (still useful for wiring).

Press 'Recalibrate' in the bottom panel to re-take reference.
"""

import sys, os, time, socket, json
import numpy as np
import pandas as pd
from zipfile import BadZipFile
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QFrame, QSizePolicy, QSystemTrayIcon,
    QTextBrowser, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, QFileSystemWatcher, QThread, pyqtSignal, QSize, QObject
from PyQt5.QtGui import QPainter, QColor, QPixmap, QFont, QIcon, QImage
from PyQt5.QtChart import (
    QChart, QChartView, QStackedBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
)

# --- Optional Windows taskbar overlay support ---
try:
    from PyQt5.QtWinExtras import QWinTaskbarButton
    WINEXTRAS_AVAILABLE = True
except Exception:
    WINEXTRAS_AVAILABLE = False

# ----------------------- OpenPose / Video imports -----------------------
# Paths — EDIT for your machine
OPENPOSE_DLLS = [
    r"C:\Users\USER\Documents\openpose_Ahmed\openpose\build\x64\Release",
    r"C:\Users\USER\Documents\openpose_Ahmed\openpose\3rdparty\windows\opencv\x64\vc15\bin",
    r"C:\Users\USER\Documents\openpose_Ahmed\openpose\3rdparty\windows\caffe\bin",
]
OPENPOSE_PY = r"C:\Users\USER\Documents\openpose_Ahmed\openpose\build\python\openpose\Release"
MODEL_FOLDER = r"C:\Users\USER\Documents\openpose_Ahmed\openpose\models"
MODEL_POSE   = "BODY_25"

for p in OPENPOSE_DLLS:
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass
if os.path.isdir(OPENPOSE_PY) and OPENPOSE_PY not in sys.path:
    sys.path.append(OPENPOSE_PY)

# Try OpenPose
OP_AVAILABLE = True
try:
    import pyopenpose as op
except Exception:
    OP_AVAILABLE = False

# Try RealSense, else webcam
RS_AVAILABLE = True
try:
    import pyrealsense2 as rs
except Exception:
    RS_AVAILABLE = False

import cv2

# ----------------------- Config (Dashboard) -----------------------
POSTURES = [
    "upright",
    "lean_right",
    "lean_left",
    "lean_forward",
    "lean_forward_right",
    "lean_forward_left",
    "lean_back",
    "lean_back_right",
    "lean_back_left",
]

REFRESH_SEC = 10
MAX_WINDOWS = 6
DEFAULT_DATA_PATH = r"\\DESKTOP-JLNTP4P\AnnotatedData\posture_data_minutes.xlsx"

# UDP listener settings
UDP_PORT = 5055
UDP_BUFSIZE = 4096
UDP_BIND_CANDIDATES = ["0.0.0.0", "127.0.0.1"]

# Optional avatar images (if missing, a placeholder is drawn)
AVATAR_PATHS = {p: "" for p in POSTURES}

# Colors for bars (distinct colors for all 9 categories)
PALETTE = {
    "upright": QColor("#4CAF50"),           # green
    "lean_right": QColor("#FF9800"),        # orange
    "lean_left": QColor("#03A9F4"),         # light blue
    "lean_forward": QColor("#F44336"),      # red
    "lean_forward_right": QColor("#8BC34A"),# light green
    "lean_forward_left": QColor("#00BCD4"), # teal
    "lean_back": QColor("#9C27B0"),         # purple
    "lean_back_right": QColor("#795548"),   # brown
    "lean_back_left": QColor("#607D8B"),    # blue gray
}

# Map incoming UDP *label strings* to internal posture keys (for color + display)
UDP_TO_POSTURE_KEY = {
    "UPRIGHT": "upright",
    "UPRIGHT RIGHT": "lean_right",
    "UPRIGHT LEFT": "lean_left",
    "FORWARD": "lean_forward",
    "FORWARD RIGHT": "lean_forward_right",
    "FORWARD LEFT": "lean_forward_left",
    "BACK": "lean_back",
    "BACK RIGHT": "lean_back_right",
    "BACK LEFT": "lean_back_left",
}


# Try LLM
try:
    from llm_api import get_analysis_data, get_chat_response
except ImportError:
    print("Error: Make sure the llm_api.py file is in the same folder")
    sys.exit()


# ---- Taskbar overlay tuning (pixels) ----
OVERLAY_BASE_CANVAS = 32      # <<< CONTROL HERE: virtual canvas (16–32 is fine)
OVERLAY_DIAMETER_PX  = 33     # <<< CONTROL HERE: circle diameter (dot size)
OVERLAY_OFFSET_PX    = (0, 0) # <<< CONTROL HERE: (x, y) nudge inside overlay square

# --------------------- Data loading ---------------------
def _postprocess_minutes(df: pd.DataFrame) -> pd.DataFrame:
    required = ["id", "start_time", "end_time"] + POSTURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df = df.sort_values(["start_time", "id"], kind="stable")

    for p in POSTURES:
        df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0).clip(lower=0)
    totals = df[POSTURES].sum(axis=1)
    overs = totals > 10.0
    if overs.any():
        df.loc[overs, POSTURES] = df.loc[overs, POSTURES].div(totals[overs], axis=0) * 10.0

    return df.tail(MAX_WINDOWS).copy()

def safe_load_minutes(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    time.sleep(0.25)

    if ext == ".csv":
        df = pd.read_csv(path)
        return _postprocess_minutes(df)

    if ext in (".xlsx", ".xlsm", ".xltx"):
        last_err = None
        for _ in range(4):
            try:
                df = pd.read_excel(path, engine="openpyxl")
                return _postprocess_minutes(df)
            except (BadZipFile, PermissionError) as e:
                last_err = e; time.sleep(0.4)
            except Exception as e:
                last_err = e; time.sleep(0.25)
        raise RuntimeError(
            f"Failed to read Excel file '{path}'. Ensure it's saved as .xlsx (not .xls) and not open/locked. "
            f"Last error: {last_err}"
        )

    if ext == ".xls":
        raise ValueError("Legacy .xls is not supported. Please save as .xlsx or export to CSV.")

    raise ValueError(f"Unsupported file type: {ext}. Use .xlsx or .csv")

# --------------------- UDP listener thread ---------------------
class UdpListener(QThread):
    error = pyqtSignal(str)
    label_received = pyqtSignal(str)

    def __init__(self, port: int, bufsize: int = 4096, parent=None, bind_candidates=None):
        super().__init__(parent)
        self.port = port
        self.bufsize = bufsize
        self._running = False
        self._sock = None
        self.bind_candidates = bind_candidates or UDP_BIND_CANDIDATES

    def _try_bind(self):
        last_err = None
        for host in self.bind_candidates:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except Exception:
                    pass
                s.bind((host, self.port))
                print(f"[UDP] Listening on {host}:{self.port} ...")
                return s
            except Exception as e:
                last_err = e
                print(f"[UDP] Bind failed on {host}:{self.port} → {e}")
                continue
        return None if last_err is None else last_err

    def run(self):
        self._running = True
        while self._running:
            sock_or_err = self._try_bind()
            if isinstance(sock_or_err, socket.socket):
                self._sock = sock_or_err
                break
            else:
                self.error.emit(f"Bind failed on all candidates: {sock_or_err}")
                time.sleep(2.0)
                if not self._running:
                    print("[UDP] Listener stopped (before bind).")
                    return

        try:
            while self._running:
                try:
                    data, addr = self._sock.recvfrom(self.bufsize)
                    if not data:
                        continue
                    msg = data.decode("utf-8", errors="ignore").strip()
                    print(f"[UDP] from {addr}: {msg}")
                    self.label_received.emit(msg)
                except OSError:
                    if not self._running:
                        break
                    print("[UDP] Socket OSError—re-binding in 1s...")
                    time.sleep(1.0)
                    try:
                        if self._sock:
                            self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                    while self._running:
                        sock_or_err = self._try_bind()
                        if isinstance(sock_or_err, socket.socket):
                            self._sock = sock_or_err
                            break
                        else:
                            self.error.emit(f"Rebind failed: {sock_or_err}")
                            time.sleep(2.0)
                except Exception as e:
                    print(f"[UDP] recv error: {e}")
                    time.sleep(0.2)
                    continue
        finally:
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None
            print("[UDP] Listener stopped.")

    def stop(self):
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

# --------------------- UI widgets (top dashboard) ---------------------
BADGE_SHAPE = "rounded"   # options: "rounded", "circle", "square", "diamond"

class LegendTile(QFrame):
    """Legend item: colored strip + avatar + posture label."""
    def __init__(self, posture: str, color: QColor, avatar_path: str = ""):
        super().__init__()
        self.setObjectName("LegendTile")
        self.posture = posture
        self.color = color
        self.avatar = self._load_avatar(avatar_path)
        self.setFixedHeight(52)
        self.setMinimumWidth(270)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame#LegendTile {
                border-radius: 10px;
                border: 2px solid rgba(0,0,0,40);
                background: rgba(255,255,255,220);
            }
        """)

    def _load_avatar(self, path: str) -> QPixmap:
        if path and os.path.exists(path):
            pm = QPixmap(path)
            if not pm.isNull():
                return pm

        size = 40
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self.color)
        p.setPen(Qt.NoPen)

        shape = getattr(self, "badge_shape", BADGE_SHAPE)
        if shape == "circle":
            p.drawEllipse(0, 0, size, size)
        elif shape == "square":
            p.drawRect(0, 0, size, size)
        elif shape == "diamond":
            from PyQt5.QtGui import QPainterPath
            path = QPainterPath()
            path.moveTo(size/2, 0)
            path.lineTo(size, size/2)
            path.lineTo(size/2, size)
            path.lineTo(0, size/2)
            path.closeSubpath()
            p.drawPath(path)
        else:  # "rounded"
            p.drawRoundedRect(0, 0, size, size, 8, 8)

        p.end()
        return pm

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        box_rect = self.rect().adjusted(8, 8, -8, -8)
        left, top = box_rect.left(), box_rect.top()
        p.setBrush(self.color); p.setPen(Qt.NoPen)
        p.drawRoundedRect(left, top, 50, box_rect.height(), 10, 10)  # colored strip
        p.drawPixmap(left + 5, top + 5, 40, 40, self.avatar)        # avatar
        p.setPen(QColor(30, 30, 30))
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(box_rect.adjusted(60, 6, -6, -6), Qt.AlignVCenter | Qt.AlignLeft, self.posture)
        p.end()

class PostureChart(QChart):
    def __init__(self):
        super().__init__()
        self.setTitle("Posture Trends — Last Hour (Minutes per 10-min window)")
        self.setAnimationOptions(QChart.SeriesAnimations)
        self._series = None
        self._categories = []

    def _build_series(self):
        s = QStackedBarSeries()
        s.setBarWidth(0.35)
        for posture in POSTURES:
            bs = QBarSet(posture)
            c = PALETTE.get(posture, QColor("gray"))
            bs.setColor(c)
            s.append(bs)
        return s

    def update_chart(self, df: pd.DataFrame):
        categories = [f"{st.strftime('%H:%M')}-{et.strftime('%H:%M')}"
                      for st, et in zip(df["start_time"], df["end_time"])]

        rebuild = (self._series is None) or (categories != self._categories)
        if rebuild:
            self.removeAllSeries()
            for axis in self.axes():
                self.removeAxis(axis)
            self._series = self._build_series()
            self.addSeries(self._series)

            ax_x = QBarCategoryAxis(); ax_x.append(categories)
            ax_y = QValueAxis(); ax_y.setRange(0, 10); ax_y.setTickCount(6)
            ax_y.setTitleText("Minutes (per 10-min window)")

            self.addAxis(ax_x, Qt.AlignBottom)
            self.addAxis(ax_y, Qt.AlignLeft)
            self._series.attachAxis(ax_x); self._series.attachAxis(ax_y)
            self.legend().setAlignment(Qt.AlignBottom)
            self._categories = categories

        for i, posture in enumerate(POSTURES):
            values = list(df[posture].values)
            bar_set = self._series.barSets()[i]
            while bar_set.count() > 0:
                bar_set.remove(0, 1)
            bar_set.append(values)

# --------------------- OpenPose worker + panel (bottom) ---------------------
BODY_25_PAIRS = [
    (0,1), (1,2), (2,3), (3,4),
    (1,5), (5,6), (6,7),
    (1,8), (8,9), (9,10), (10,11),
    (8,12), (12,13), (13,14),
    (0,15), (15,17),
    (0,16), (16,18),
    (14,19), (19,20), (14,21),
    (11,22), (22,23), (11,24)
]

def draw_skeleton(canvas, pts2d, color=(0,0,0), kpr=3, thick=2):
    H, W = canvas.shape[:2]
    valid = ~np.isnan(pts2d[:,0])
    for a, b in BODY_25_PAIRS:
        if a < len(pts2d) and b < len(pts2d) and valid[a] and valid[b]:
            pa = tuple(np.round(pts2d[a]).astype(int))
            pb = tuple(np.round(pts2d[b]).astype(int))
            if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
                cv2.line(canvas, pa, pb, color, thick, cv2.LINE_AA)
    for i in range(min(len(pts2d), 25)):
        if valid[i]:
            p = tuple(np.round(pts2d[i]).astype(int))
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(canvas, p, kpr, color, -1, cv2.LINE_AA)

def average_keypoints(stack):
    arr = np.stack(stack, axis=0)  # [N,25,2]
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(arr, axis=0)  # [25,2]
    return mean

def extract_person0_2d(datum):
    if not OP_AVAILABLE:
        return None
    if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
        return None
    kp = datum.poseKeypoints[0]  # [25,3]
    pts = kp[:, :2].astype(np.float32)
    conf = kp[:, 2]
    pts[conf < 0.05] = np.nan
    return pts

def qimage_from_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return qimg.copy()

class OpenPoseWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    status = pyqtSignal(str)
    op_ready = pyqtSignal(bool)

    def __init__(self, target_fps=24, calib_frames=50, parent=None):
        super().__init__(parent)
        self.target_fps = max(1, int(target_fps))
        self.calib_frames = max(1, int(calib_frames))
        self._running = False
        self._do_recalib = False
        self._ref_skeleton = None

        self._pipeline = None
        self._cap = None

        self._op_wrapper = None
        self._op_available = OP_AVAILABLE
        self._rs_available = RS_AVAILABLE

    def stop(self):
        self._running = False

    def request_recalibration(self):
        self._do_recalib = True

    def _init_video(self):
        # Try RealSense color stream
        if self._rs_available:
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(cfg)
                self._pipeline = pipeline
                self.status.emit("[Video] RealSense started.")
                return
            except Exception as e:
                self.status.emit(f"[Video] RealSense failed ({e}), fallback to webcam.")
                self._pipeline = None

        # Webcam fallback
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status.emit("ERROR: No camera available.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap = cap
        self.status.emit("[Video] Webcam started.")

    def _grab_frame(self):
        if self._pipeline is not None:
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                return None
            return np.asanyarray(color.get_data())
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

    def _close_video(self):
        if self._pipeline is not None:
            try: self._pipeline.stop()
            except Exception: pass
            self._pipeline = None
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    def _init_openpose(self):
        if not self._op_available:
            self.op_ready.emit(False)
            self.status.emit("[OpenPose] Not available; showing raw frames.")
            return
        try:
            params = {
                "model_folder": MODEL_FOLDER,
                "model_pose": MODEL_POSE,
                "number_people_max": 1,
                "net_resolution": "-1x256",
            }
            self._op_wrapper = op.WrapperPython()
            self._op_wrapper.configure(params)
            self._op_wrapper.start()
            self.op_ready.emit(True)
            self.status.emit("[OpenPose] Started.")
        except Exception as e:
            self._op_wrapper = None
            self._op_available = False
            self.op_ready.emit(False)
            self.status.emit(f"[OpenPose] Failed to start ({e}); showing raw frames.")

    def _run_openpose(self, frame_bgr):
        if not self._op_available or self._op_wrapper is None:
            return None
        try:
            datum = op.Datum()
            datum.cvInputData = frame_bgr
            # robust emplaceAndPop
            try:
                datums = op.VectorDatum(); datums.append(datum)
                self._op_wrapper.emplaceAndPop(datums)
            except Exception:
                self._op_wrapper.emplaceAndPop([datum])
            return datum
        except Exception:
            return None

    def _calibrate(self):
        self.status.emit(f"[Calib] Hold still... collecting {self.calib_frames} frames.")
        ref_samples = []
        while self._running and len(ref_samples) < self.calib_frames:
            frame = self._grab_frame()
            if frame is None:
                continue
            datum = self._run_openpose(frame)
            pts = extract_person0_2d(datum) if datum is not None else None

            # live preview during calibration
            canvas = np.full_like(frame, 255)
            if pts is not None:
                draw_skeleton(canvas, pts, (140,140,255), kpr=2, thick=1)
            self.frame_ready.emit(qimage_from_bgr(canvas))

            if pts is not None:
                ref_samples.append(pts)

            # keep a reasonable cadence
            cv2.waitKey(1)

        if ref_samples:
            self._ref_skeleton = average_keypoints(ref_samples)
            self.status.emit("[Calib] Reference skeleton fixed.")
        else:
            self._ref_skeleton = None
            self.status.emit("[Calib] Failed (no keypoints). Try again.")

    def run(self):
        self._running = True
        self._init_video()
        self._init_openpose()

        # initial calibration only if OpenPose is active
        if self._op_available and self._op_wrapper is not None:
            self._calibrate()

        min_dt = 1.0 / float(max(1, self.target_fps))
        REF_COLOR  = (0, 200, 0)
        LIVE_COLOR = (0,   0, 0)
        KP_RADIUS_REF, KP_RADIUS_LIVE = 4, 3
        THICK_REF, THICK_LIVE = 4, 2

        try:
            while self._running:
                t0 = time.time()
                if self._do_recalib and self._op_available and self._op_wrapper is not None:
                    self._do_recalib = False
                    self._calibrate()

                frame = self._grab_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                canvas = np.full_like(frame, 255)

                if self._op_available and self._op_wrapper is not None:
                    datum = self._run_openpose(frame)
                    pts_live = extract_person0_2d(datum) if datum is not None else None
                    if self._ref_skeleton is not None:
                        draw_skeleton(canvas, self._ref_skeleton, REF_COLOR, kpr=KP_RADIUS_REF, thick=THICK_REF)
                    if pts_live is not None:
                        draw_skeleton(canvas, pts_live, LIVE_COLOR, kpr=KP_RADIUS_LIVE, thick=THICK_LIVE)
                else:
                    # no OpenPose → show raw frames
                    canvas = frame

                self.frame_ready.emit(qimage_from_bgr(canvas))

                dt = time.time() - t0
                if dt < min_dt:
                    time.sleep(min_dt - dt)
        finally:
            self._close_video()

# ---- Bottom panel widget ----
class OpenPosePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #ddd; border-radius: 8px; }")

        title = QLabel("Live Skeleton (OpenPose / RealSense)")
        title.setStyleSheet("QLabel { font-size: 11pt; font-weight: 600; }")

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(QSize(320, 360))
        self.video_label.setStyleSheet("QLabel { background: #f7f7f7; border: 1px solid #eee; }")

        self.status_label = QLabel("—")
        self.status_label.setStyleSheet("QLabel { color: #555; }")

        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_recal = QPushButton("Recalibrate")

        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self.btn_start)
        btn_bar.addWidget(self.btn_stop)
        btn_bar.addWidget(self.btn_recal)
        btn_bar.addStretch(1)
        btn_bar.addWidget(self.status_label)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8,8,8,8)
        lay.addWidget(title)
        lay.addWidget(self.video_label, 1)
        lay.addLayout(btn_bar)

        self.worker = None
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_recal.clicked.connect(self.recalibrate)

    def start_worker(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = OpenPoseWorker(target_fps=24, calib_frames=50, parent=self)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.op_ready.connect(self._on_op_ready)
        self.worker.start()
        self._on_status("Worker started.")

    def stop_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
            self._on_status("Worker stopped.")
        self.worker = None

    def recalibrate(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_recalibration()
            self._on_status("Recalibration requested.")

    def _on_frame(self, qimg: QImage):
        # Fit into label while preserving aspect ratio
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_status(self, msg: str):
        self.status_label.setText(msg)

    def _on_op_ready(self, ok: bool):
        self._on_status("OpenPose ready." if ok else "OpenPose not available, showing camera frames.")

    def closeEvent(self, event):
        self.stop_worker()
        super().closeEvent(event)

# ---- Collapsible Section Widget ----
class CollapsibleSection(QFrame):
    TRUNCATE_LIMIT = 100 

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)

        self._is_expanded = False
        self._full_text = ""

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #333;")

        self.view_all_button = QPushButton("Read more")
        self.view_all_button.setStyleSheet("QPushButton { border: none; color: #007BFF; font-size: 9pt; }")
        self.view_all_button.setCursor(Qt.PointingHandCursor)
        self.view_all_button.setFlat(True)
        self.view_all_button.hide()

        self.text_browser = QTextBrowser()
        self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
        """)

        title_layout = QHBoxLayout()
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.view_all_button)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        main_layout.addLayout(title_layout)
        main_layout.addWidget(self.text_browser)

        self.view_all_button.clicked.connect(self.toggle_view)

    def setText(self, text: str):
        self._full_text = text
        
        if len(text) > self.TRUNCATE_LIMIT:
            truncated_text = text[:self.TRUNCATE_LIMIT] + "..."
            self.text_browser.setText(truncated_text)
            self.view_all_button.show()
            
            if self._is_expanded:
                self.toggle_view()
            else:
                self.text_browser.setFixedHeight(int(self.text_browser.document().size().height()) + 10)

        else:
            self.text_browser.setText(text)
            self.view_all_button.hide()
            self.text_browser.setFixedHeight(int(self.text_browser.document().size().height()) + 10)

    def toggle_view(self):
        if self._is_expanded:
            truncated_text = self._full_text[:self.TRUNCATE_LIMIT] + "..."
            self.text_browser.setText(truncated_text)
            self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.text_browser.setFixedHeight(int(self.text_browser.document().size().height()) + 10)
            self.view_all_button.setText("Read more")
            self._is_expanded = False
        else:
            self.text_browser.setText(self._full_text)
            self.text_browser.setMinimumHeight(0)
            self.text_browser.setMaximumHeight(16777215) 
            self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.view_all_button.setText("Close")
            self._is_expanded = True


# ---- Bottom Left panel widget ----
class ChatModuleWidget(QFrame):
    send_chat_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.initUI() 

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        title_label = QLabel("🪑 Smart Posture Mate")
        title_label.setStyleSheet("""
            font-size: 13pt;
            font-weight: 900;
            color: #2c3e50;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        """)
        title_label.setAlignment(Qt.AlignCenter)

        # --- 2. 생성한 제목을 레이아웃의 맨 위에 추가 ---
        layout.addWidget(title_label)


        self.do_section = CollapsibleSection("✅ What to do")
        self.summary_section = CollapsibleSection("📊 Summary (last 10min)")
        self.rec_section = CollapsibleSection("💡 Recommendation")

        layout.addWidget(self.do_section)
        layout.addWidget(self.summary_section)
        layout.addWidget(self.rec_section)
        
        chat_title_label = QLabel("💬 Chat")
        chat_title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #333;")
        layout.addWidget(chat_title_label)
        
        self.chat_history_browser = QTextBrowser()
        self.chat_history_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
            }
        """)
        layout.addWidget(self.chat_history_browser, 1)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask questions about your posture.")
        self.send_button = QPushButton("Send")
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        self.send_button.clicked.connect(self._on_send_clicked)
        self.chat_input.returnPressed.connect(self._on_send_clicked)

    def _on_send_clicked(self):
        message = self.chat_input.text().strip()
        if message:
            user_html = f"<p class='user'><b>You:</b> {message}</p>"
            self.chat_history_browser.append(user_html)
            self.chat_input.clear()
            self.send_chat_message.emit(message)
    
    def add_ai_message(self, message: str):
        ai_html = f"<p class='ai'><b>AI:</b> {message}</p>"
        self.chat_history_browser.append(ai_html)


class LLMWorker(QObject):
    analysis_ready = pyqtSignal(dict)
    chat_ready = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_history = []

    def run_analysis(self):
        result = get_analysis_data()
        self.analysis_ready.emit(result)
        
    def run_chat(self, user_message: str):
        result = get_chat_response(user_message, self.chat_history)
        
        self.chat_history.append({'role': 'user', 'content': user_message})
        if 'answer' in result:
            self.chat_history.append({'role': 'assistant', 'content': result['answer']})
            
        self.chat_ready.emit(result)


# --------------------- Main window ---------------------
class MainWindow(QWidget):
    request_analysis_signal = pyqtSignal(str)
    request_chat_signal = pyqtSignal(str)

    def __init__(self, data_path: str):
        super().__init__()
        self.setWindowTitle("Posture Trends + OpenPose Live — QtCharts Pro (PyQt5)")
        self.resize(1180, 980)
        self.excel_path = data_path
        self.df = pd.DataFrame()
        self.last_seen_id = None

        # System tray (create first, but don't show until icon set)
        self.tray = QSystemTrayIcon(self)

        # Status icons (tray/window)
        self.icon_ok = self._make_icon(PALETTE["upright"])
        self.icon_bad = self._make_icon(QColor("#E53935"))
        self.icon_unknown = self._make_icon(QColor("#9E9E9E"))

        # Overlay icons (taskbar overlay) — uses size/offset knobs above
        self.overlay_ok      = self._make_overlay_icon(PALETTE["upright"])
        self.overlay_bad     = self._make_overlay_icon(QColor("#E53935"))
        self.overlay_unknown = self._make_overlay_icon(QColor("#9E9E9E"))

        # Prepare Windows taskbar overlay; created after window is shown
        self._taskbar_btn = None

        # Set default icons BEFORE showing tray
        self._update_status_icons(None)       # sets tray+window icon (unknown state)
        self.tray.setToolTip("Posture: unknown")
        self.tray.setVisible(True)

        # Connect tray events
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.messageClicked.connect(self._on_tray_message_clicked)

        # Header
        self.path_label = QLabel(self.excel_path or "(choose a file)")
        self.refresh_btn = QPushButton("Refresh now")
        self.pick_btn = QPushButton("Choose data file…")
        self.status_label = QLabel("—")

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Data file:"))
        hdr.addWidget(self.path_label, 1)
        hdr.addWidget(self.pick_btn)
        hdr.addWidget(self.refresh_btn)

        # Chart + legend (TOP)
        self.chart = PostureChart()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)

        self.current_label_title = QLabel("Current Sitting Posture")
        self.current_label_title.setAlignment(Qt.AlignLeft)
        self.current_label_title.setStyleSheet("QLabel { font-size: 11pt; font-weight: 600; }")

        self.current_label_box = QLabel("—")
        self.current_label_box.setAlignment(Qt.AlignCenter)
        self.current_label_box.setFixedHeight(40)
        self.current_label_box.setStyleSheet("""
            QLabel {
                border: 3px solid gray;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                background-color: white;
            }
        """)

        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.StyledPanel)
        chart_frame.setStyleSheet("QFrame { background: #fafafa; border: 1px solid #ddd; border-radius: 8px; }")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.addWidget(self.chart_view)
        chart_layout.addSpacing(6)
        chart_layout.addWidget(self.current_label_title, alignment=Qt.AlignLeft)
        chart_layout.addWidget(self.current_label_box)

        legend_panel = QFrame()
        legend_panel.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; }")
        leg = QVBoxLayout(legend_panel)
        leg.setContentsMargins(8, 8, 8, 8)
        leg.addWidget(QLabel("<b>Postures</b>"))
        for p in POSTURES:
            tile = LegendTile(p, PALETTE.get(p, QColor("gray")), AVATAR_PATHS.get(p, ""))
            leg.addWidget(tile)
        leg.addStretch(1)

        top_row = QHBoxLayout()
        top_row.addWidget(chart_frame, 3)
        top_row.addWidget(legend_panel, 1)

        # ----------------- Bottom: OpenPose panel, Chat Module -----------------
        self.op_panel = OpenPosePanel()
        self.chat_module = ChatModuleWidget()

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.op_panel, 2)
        bottom_layout.addWidget(self.chat_module, 3)

        # Root layout
        root = QVBoxLayout(self)
        root.addLayout(hdr)
        root.addLayout(top_row, 2)
        root.addWidget(self.status_label)
        root.addSpacing(6)
        root.addLayout(bottom_layout, 3)

        # Signals
        self.refresh_btn.clicked.connect(self.update_chart)
        self.pick_btn.clicked.connect(self.pick_data_file)

        # File watcher
        self.file_watcher = QFileSystemWatcher(self)
        self.file_watcher.fileChanged.connect(self._on_file_changed)
        if self.excel_path and os.path.exists(self.excel_path):
            self.file_watcher.addPath(self.excel_path)

        # Fallback timer for chart refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(REFRESH_SEC * 1000)

        # Initial load (if present)
        if self.excel_path and os.path.exists(self.excel_path):
            QTimer.singleShot(300, self.update_chart)

        # Auto-attach when file appears
        self._init_missing_file_watch()

        # UDP listener
        self.udp_thread = UdpListener(UDP_PORT, UDP_BUFSIZE, self, UDP_BIND_CANDIDATES)
        self.udp_thread.error.connect(self._on_udp_error)
        self.udp_thread.label_received.connect(self._on_udp_label)
        self.udp_thread.start()


        # LLM
        self.setup_llm_thread()
        self.chat_module.send_chat_message.connect(self.handle_chat_send)
        self.setup_periodic_analysis_timer()

    def setup_llm_thread(self):
        """Initializes the QThread and LLMWorker, and connects signals and slots."""
        self.llm_thread = QThread()
        self.llm_worker = LLMWorker()
        self.llm_worker.moveToThread(self.llm_thread)

        # Connect signals from the worker to slots in the main thread (GUI)
        self.llm_worker.analysis_ready.connect(self.update_analysis_posture)
        self.llm_worker.chat_ready.connect(self.update_chat_display)

        # Connect signals from the main thread (GUI) to slots in the worker
        self.request_analysis_signal.connect(self.llm_worker.run_analysis)
        self.request_chat_signal.connect(self.llm_worker.run_chat)
        
        self.llm_thread.start()
        print("[LLM] Worker thread started.")

    def setup_periodic_analysis_timer(self):
        """Sets up a QTimer to request posture analysis periodically."""
        self.analysis_timer = QTimer(self)
        # 10 minutes (600,000ms). Set to 15 seconds (15000) for testing.
        self.analysis_timer.setInterval(15000)
        self.analysis_timer.timeout.connect(self.request_periodic_analysis)
        self.analysis_timer.start()
        # Trigger the first analysis 1 second after startup.
        QTimer.singleShot(1000, self.request_periodic_analysis) 

    def request_periodic_analysis(self):
        """Slot called by the timer to trigger a posture analysis."""
        print("\n--- Requesting periodic posture analysis (data-independent) ---")
        self.request_analysis_signal.emit("")

    def handle_chat_send(self, message: str):
        """Slot to handle sending a user's chat message to the LLM worker."""
        self.request_chat_signal.emit(message)

    # Note: Corrected the duplicated method names from the original snippet.
    # This slot handles the CHAT response.
    def update_chat_display(self, result: dict):
        """Slot to update the chat history with the LLM's response."""
        if 'error' in result:
            self.chat_module.add_ai_message(f"Error: {result['error']}")
        else:
            self.chat_module.add_ai_message(result.get('answer', '...'))

    # Ensure Windows taskbar overlay targets this window once it's shown
    def showEvent(self, event):
        super().showEvent(event)
        if WINEXTRAS_AVAILABLE and self._taskbar_btn is None and self.windowHandle() is not None:
            try:
                self._taskbar_btn = QWinTaskbarButton(self)
                self._taskbar_btn.setWindow(self.windowHandle())
                self._update_taskbar_overlay(None)  # start grey
            except Exception:
                self._taskbar_btn = None

    # ---------- Auto-attach helpers ----------
    def _init_missing_file_watch(self):
        self._file_poll_timer = QTimer(self)
        self._file_poll_timer.timeout.connect(self._try_attach_missing_file)
        self._file_poll_timer.start(2000)

        self._dir_watcher = QFileSystemWatcher(self)
        parent_dir = os.path.dirname(self.excel_path) or "."
        if os.path.isdir(parent_dir):
            try:
                self._dir_watcher.addPath(parent_dir)
                self._dir_watcher.directoryChanged.connect(self._on_directory_changed)
            except Exception:
                pass

    def _on_directory_changed(self, _path: str):
        self._try_attach_missing_file()

    def _try_attach_missing_file(self):
        if self.excel_path and os.path.exists(self.excel_path):
            if self.excel_path not in self.file_watcher.files():
                try:
                    self.file_watcher.addPath(self.excel_path)
                except Exception:
                    pass
            self.path_label.setText(self.excel_path)
            try:
                self._file_poll_timer.stop()
            except Exception:
                pass
            QTimer.singleShot(300, self.update_chart)

    # ---------- Bring-to-front helpers ----------
    def _force_top_temporarily(self):
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show(); self.raise_(); self.activateWindow()
        QTimer.singleShot(250, self._unset_always_on_top)

    def _unset_always_on_top(self):
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show(); self.raise_(); self.activateWindow()

    def _bring_to_front(self):
        if self.isMinimized():
            self.showNormal()
        self._force_top_temporarily()

    # ---------- File watcher ----------
    def _on_file_changed(self, path):
        if os.path.exists(path) and path not in self.file_watcher.files():
            try: self.file_watcher.addPath(path)
            except Exception: pass
        QTimer.singleShot(300, self.update_chart)

    # ---------- Notifications ----------
    def _notify(self, title: str, message: str, msec: int = 8000, _retry=True):
        if self.tray.isVisible():
            self.tray.showMessage(title, message, QSystemTrayIcon.Information, msec)
        elif _retry:
            QTimer.singleShot(300, lambda: self._notify(title, message, msec, _retry=False))

    def _on_tray_message_clicked(self):
        self._bring_to_front()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self._bring_to_front()

    def _on_udp_error(self, msg: str):
        print(f"[UDP] ERROR: {msg}")

    # --- Handle UDP message: update label, border color, tray + taskbar indicators ---
    def _on_udp_label(self, raw_msg: str):
        """
        Expects UDP payload in this format:
        {'type': 'label', 'label': 'FORWARD', 'frame': 901, 'ts': '2025-08-12 00:55:37.198', 'session_start': '2025-08-12 00:54:35', 'warning': True}
        """
        label_text = raw_msg.strip()
        warning_flag = False  # default

        # Parse JSON-ish dict; tolerate both JSON and Python-dict-like strings
        try:
            if label_text.startswith("{") and "'" in label_text and '"label"' not in label_text:
                label_text_json = label_text.replace("'", '"')
            else:
                label_text_json = label_text
            obj = json.loads(label_text_json)
            if isinstance(obj, dict):
                if "label" in obj:
                    label_text = str(obj["label"])
                warning_flag = bool(obj.get("warning", False))
        except Exception:
            pass

        label_upper = label_text.upper().strip()
        ts_now = time.strftime("%H:%M:%S")

        key = UDP_TO_POSTURE_KEY.get(label_upper)
        if key:
            display_text = key
            color = PALETTE.get(key, QColor("gray"))
            border_css = (
                f"border: 3px solid {color.name()};"
                "border-radius: 8px; font-size: 14pt; font-weight: bold; background-color: white;"
            )
            self.current_label_box.setStyleSheet(f"QLabel {{{border_css}}}")
        else:
            display_text = label_upper
            self.current_label_box.setStyleSheet("""
                QLabel {
                    border: 3px solid gray;
                    border-radius: 8px;
                    font-size: 14pt;
                    font-weight: bold;
                    background-color: white;
                }
            """)

        self.current_label_box.setText(f"{ts_now} — {display_text}")

        # Update tray/window icon + Windows taskbar overlay
        if key is None:
            self._update_status_icons(None)
            self.tray.setToolTip(f"Posture: {label_upper}")
        else:
            is_ok = (key == "upright")
            self._update_status_icons(is_ok)
            self.tray.setToolTip(f"Posture: {key}")

        # 🚨 Aggressive warning if bad posture detected
        if warning_flag:
            self._show_aggressive_warning(display_text)

    def _show_aggressive_warning(self, posture_name: str):
        QApplication.beep()
        self._bring_to_front()

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("⚠ Posture Alert!")
        msg.setText(f"You are in a bad posture: {posture_name.upper()}.\n\nPlease correct your posture immediately!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setStyleSheet("""
            QMessageBox { font-size: 16pt; font-weight: bold; }
            QLabel { font-size: 14pt; }
            QPushButton { font-size: 12pt; padding: 8px 16px; min-width: 90px; }
        """)
        msg.exec_()

    # ---------- App behavior ----------
    def pick_data_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data file", "",
            "Data Files (*.xlsx *.xlsm *.xltx *.csv);;Excel Files (*.xlsx *.xlsm *.xltx);;CSV Files (*.csv)"
        )
        if path:
            self.excel_path = path
            self.path_label.setText(path)
            self.last_seen_id = None
            try: self.file_watcher.removePaths(self.file_watcher.files())
            except Exception: pass
            self.file_watcher.addPath(path)
            QTimer.singleShot(300, self.update_chart)

    def update_chart(self):
        if not self.excel_path or not os.path.exists(self.excel_path):
            if self.sender() == self.refresh_btn:
                QMessageBox.warning(self, "File not found", f"Cannot find:\n{self.excel_path}")
            return
        try:
            new_df = safe_load_minutes(self.excel_path)
            self.df = new_df
            self.chart.update_chart(self.df)
            self.status_label.setText("Updated ✔")

            new_last_id = int(new_df["id"].max()) if not new_df.empty else None
            should_notify = new_last_id is not None and (
                self.last_seen_id is None or new_last_id > self.last_seen_id
            )
            if should_notify:
                windows_shown = len(new_df)
                msg = f"{windows_shown} window(s) on chart. Last ID: {new_last_id}"
                self._notify("Posture summary updated", msg)

            self.last_seen_id = new_last_id

        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    # ---------- Icon helpers ----------
    def _make_icon(self, color: QColor) -> QIcon:
        icon = QIcon()
        for size in (16, 20, 24, 32):
            pm = QPixmap(size, size)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, size, size)
            p.end()
            icon.addPixmap(pm)
        return icon

    def _make_overlay_icon(self, color: QColor,
                           base=OVERLAY_BASE_CANVAS,
                           diam=OVERLAY_DIAMETER_PX,
                           offset=OVERLAY_OFFSET_PX) -> QIcon:
        """
        Build a taskbar overlay icon (transparent square with a colored dot).
        Windows anchors this overlay at the bottom-right of the taskbar button.
        'diam' controls dot size; 'offset' nudges the dot within the overlay square.
        """
        icon = QIcon()
        for canvas in (16, 20, 24, 32):
            scale = canvas / float(base)
            d  = max(1, int(diam * scale))
            ox = int(offset[0] * scale)
            oy = int(offset[1] * scale)

            pm = QPixmap(canvas, canvas)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(color)
            p.setPen(Qt.NoPen)

            x = (canvas - d) // 2 + ox
            y = (canvas - d) // 2 + oy
            x = max(0, min(canvas - d, x))
            y = max(0, min(canvas - d, y))

            p.drawEllipse(x, y, d, d)
            p.end()
            icon.addPixmap(pm)
        return icon

    def _update_status_icons(self, ok):
        icon = self.icon_unknown if ok is None else (self.icon_ok if ok else self.icon_bad)
        self.tray.setIcon(icon)
        self.setWindowIcon(icon)
        self._update_taskbar_overlay(ok)

    def _update_taskbar_overlay(self, ok):
        if not (WINEXTRAS_AVAILABLE and self._taskbar_btn):
            return
        icon = self.overlay_unknown if ok is None else (self.overlay_ok if ok else self.overlay_bad)
        try:
            self._taskbar_btn.setOverlayIcon(icon)
        except Exception:
            pass
    
    def update_analysis_posture(self, result: dict):
        if 'error' in result:
            print(f"[LLM Error] Analysis: {result['error']}")
            self.chat_module.summary_section.setText(f"Error: {result['error']}")
            return

        self.chat_module.do_section.setText(result.get('what_to_do', ''))
        self.chat_module.summary_section.setText(result.get('summary', ''))
        self.chat_module.rec_section.setText(result.get('recommendation', ''))
        print("[LLM] Analysis updated.")

    # ---------- Cleanup ----------
    def closeEvent(self, event):
        try:
            if hasattr(self, "udp_thread") and self.udp_thread.isRunning():
                self.udp_thread.stop()
                self.udp_thread.wait(1500)
        except Exception:
            pass
        try:
            if hasattr(self, "op_panel"):
                self.op_panel.stop_worker()
        except Exception:
            pass
        return super().closeEvent(event)

# --------------------- Entry point ---------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow(DEFAULT_DATA_PATH)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
