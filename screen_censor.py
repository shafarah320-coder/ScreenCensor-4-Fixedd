import os
import sys
import time
import threading
import tempfile
import ctypes
from ctypes import wintypes
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import mss
from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont
from nudenet import NudeDetector

APP_TITLE = "Screen Censor 4.7"
# NudeNet classes used for breast/genital-area censoring. Covered labels are
# supported only if the bundled model actually emits them.
CENSOR_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
}
LABEL_THRESHOLDS = {
    "FEMALE_BREAST_EXPOSED": 0.28,
    "FEMALE_BREAST_COVERED": 0.35,
    "FEMALE_GENITALIA_EXPOSED": 0.28,
    "FEMALE_GENITALIA_COVERED": 0.35,
}
SMOOTHING_ALPHA = 0.52  # balances responsiveness and stable movement
DETECTION_INTERVAL = 0.15      # Tracking updates about 6-7 times/sec; actual speed depends on PC/model.
OVERLAY_REFRESH_MS = 16        # UI refresh target: about 60 Hz; not 60 AI detections/sec.
BOX_PADDING = 0.22
BLUR_RADIUS = 15
TILE_ROWS = 2
TILE_COLS = 2
TILE_OVERLAP = 0.18

IS_WINDOWS = sys.platform == "win32"
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_F12 = 0x7B
WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND), ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD), ("pt", POINT)
    ]

class ScreenCensor:
    def __init__(self, root):
        if not IS_WINDOWS:
            raise RuntimeError("Screen Censor 4.0 Desktop Overlay hanya untuk Windows 10/11.")
        self.root = root
        root.title(APP_TITLE)
        root.geometry("500x455")
        root.resizable(False, False)
        root.configure(bg="#151923")
        root.attributes("-topmost", True)

        self.running = False
        self.closing = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.boxes = []
        self.patch_data = []
        self.patch_images = []
        self.patch_version = 0
        self.rendered_patch_version = -1
        self.sensor_mode = tk.StringVar(value="Blur + LOOSER")
        self.scan_target = tk.StringVar(value="Semua target")
        self.custom_text = tk.StringVar(value="LOOSER")
        self.text_color = tk.StringVar(value="Putih")
        self.full_blur_image = None
        self.detector = None
        self.worker = None
        self.overlay_hwnd = None
        self.last_error = None
        self.frame_size = (0, 0)
        self.previous_boxes = []
        self.track_records = []  # short-lived motion tracks smooth brief detection gaps
        self.hotkey_registered = False

        tk.Label(root, text="SCREEN CENSOR 4.7", fg="white", bg="#151923",
                 font=("Segoe UI", 17, "bold")).pack(pady=(16, 3))
        tk.Label(root, text="Desktop blur overlay • wajah + area tubuh terdeteksi model",
                 fg="#b9c2d0", bg="#151923", font=("Segoe UI", 9)).pack()

        self.status = tk.Label(root, text="Status: Mati", fg="#ff7777",
                               bg="#151923", font=("Segoe UI", 10, "bold"))
        self.status.pack(pady=12)

        row = tk.Frame(root, bg="#151923")
        row.pack()
        self.start_btn = tk.Button(row, text="MULAI SENSOR", width=16,
                                   command=self.start, bg="#238b57", fg="white",
                                   relief="flat", font=("Segoe UI", 10, "bold"))
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = tk.Button(row, text="MATIKAN", width=12,
                                  command=self.stop, state="disabled",
                                  bg="#a83e45", fg="white", relief="flat",
                                  font=("Segoe UI", 10, "bold"))
        self.stop_btn.pack(side="left", padx=5)

        mode_row = tk.Frame(root, bg="#151923")
        mode_row.pack(pady=(2, 0))
        tk.Label(mode_row, text="Tipe sensor:", fg="#b9c2d0", bg="#151923",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        self.mode_menu = tk.OptionMenu(
            mode_row, self.sensor_mode,
            "Blur + LOOSER", "LOOSER tunggal", "TV semut", "Blur semua + fokus",
            "Kotak hitam", "Pixelate", "Glitch", "Teks custom", "Emoji cover",
            command=self._change_sensor_mode
        )
        self.mode_menu.config(bg="#283244", fg="white", activebackground="#35445c",
                              activeforeground="white", relief="flat", highlightthickness=0)
        self.mode_menu["menu"].config(bg="#283244", fg="white")
        self.mode_menu.pack(side="left")

        target_row = tk.Frame(root, bg="#151923")
        target_row.pack(pady=(10, 0))
        tk.Label(target_row, text="Scan bagian:", fg="#b9c2d0", bg="#151923",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        self.target_menu = tk.OptionMenu(
            target_row, self.scan_target,
            "Semua target", "Dada saja", "Wajah saja", "Bagian intim saja",
            command=self._change_scan_target
        )
        self.target_menu.config(bg="#283244", fg="white", activebackground="#35445c",
                                activeforeground="white", relief="flat", highlightthickness=0)
        self.target_menu["menu"].config(bg="#283244", fg="white")
        self.target_menu.pack(side="left")

        text_row = tk.Frame(root, bg="#151923")
        text_row.pack(fill="x", padx=18, pady=(10, 0))
        tk.Label(text_row, text="Teks sensor:", fg="#b9c2d0", bg="#151923",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 7))
        self.custom_entry = tk.Entry(text_row, textvariable=self.custom_text, width=19,
                                     bg="#283244", fg="white", insertbackground="white",
                                     relief="flat")
        self.custom_entry.pack(side="left", padx=(0, 8), ipady=4)
        self.custom_entry.bind("<KeyRelease>", self._custom_text_changed)
        tk.Label(text_row, text="Warna:", fg="#b9c2d0", bg="#151923",
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 5))
        self.color_menu = tk.OptionMenu(
            text_row, self.text_color, "Putih", "Merah", "Kuning", "Cyan", "Hijau",
            command=self._change_text_color
        )
        self.color_menu.config(bg="#283244", fg="white", activebackground="#35445c",
                               activeforeground="white", relief="flat", highlightthickness=0)
        self.color_menu["menu"].config(bg="#283244", fg="white")
        self.color_menu.pack(side="left")

        tk.Label(root, text="Blur semua + fokus: layar diblur, area target terdeteksi tetap jelas.",
                 fg="#9aa5b5", bg="#151923", font=("Segoe UI", 8)).pack(pady=(8, 0))
        tk.Label(root, text="Preset: blur • teks • TV semut • kotak hitam • pixelate • glitch • emoji",
                 fg="#9aa5b5", bg="#151923", font=("Segoe UI", 8)).pack(pady=(5, 0))
        tk.Label(root, text="Tombol darurat global: Ctrl + Shift + F12",
                 fg="#ffd166", bg="#151923", font=("Segoe UI", 9, "bold")).pack(pady=(10, 3))
        tk.Label(root, text="Overlay desktop di atas aplikasi lain • LOOSER menutup area mata",
                 fg="#9aa5b5", bg="#151923", font=("Segoe UI", 8)).pack()
        tk.Label(root, text="Deteksi wajah berlaku untuk semua wajah; aplikasi tidak menebak gender.",
                 fg="#9aa5b5", bg="#151923", font=("Segoe UI", 8)).pack(pady=(3, 0))

        self.overlay = tk.Toplevel(root)
        self.overlay.withdraw()
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg="magenta")
        self.overlay.wm_attributes("-transparentcolor", "magenta")
        self.canvas = tk.Canvas(self.overlay, bg="magenta", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._register_hotkey)
        self.root.after(OVERLAY_REFRESH_MS, self.refresh_overlay)
        self.root.bind_all("<Control-Shift-F12>", lambda _e: self.stop())

    def _register_hotkey(self):
        if not self.closing:
            try:
                self.hotkey_registered = bool(ctypes.windll.user32.RegisterHotKey(
                    None, 0x5343, MOD_CONTROL | MOD_SHIFT, VK_F12))
            except Exception:
                self.hotkey_registered = False
            self.root.after(35, self._poll_hotkey)

    def _poll_hotkey(self):
        if self.closing:
            return
        if self.hotkey_registered:
            msg = MSG()
            while ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None,
                                                     WM_HOTKEY, WM_HOTKEY, PM_REMOVE):
                self.root.after(0, self.stop)
        self.root.after(35, self._poll_hotkey)

    def _configure_overlay(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        # Keep the overlay hidden until the Windows color-key transparency
        # has been configured. This prevents an opaque full-screen window
        # from briefly covering the desktop if transparency setup fails.
        self.overlay.withdraw()
        self.overlay.geometry(f"{sw}x{sh}+0+0")
        self.overlay.update_idletasks()

        hwnd_child = wintypes.HWND(self.overlay.winfo_id())
        if not hwnd_child:
            raise RuntimeError("Windows tidak memberikan handle untuk jendela overlay.")

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Tk may return the client-area HWND. Mouse transparency must be set
        # on the actual top-level window so clicks/drags reach the desktop apps.
        get_ancestor = user32.GetAncestor
        get_ancestor.argtypes = [wintypes.HWND, wintypes.UINT]
        get_ancestor.restype = wintypes.HWND
        GA_ROOT = 2
        hwnd = get_ancestor(hwnd_child, GA_ROOT) or hwnd_child

        get_long_ptr = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long_ptr = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int]
        get_long_ptr.restype = ctypes.c_ssize_t
        set_long_ptr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_long_ptr.restype = ctypes.c_ssize_t

        ctypes.set_last_error(0)
        style = get_long_ptr(hwnd, GWL_EXSTYLE)
        if ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())

        ctypes.set_last_error(0)
        set_long_ptr(
            hwnd, GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        )
        if ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())

        # Explicitly apply the magenta color key. Tk's -transparentcolor
        # setting alone may not survive changes to extended window styles.
        set_layered = user32.SetLayeredWindowAttributes
        set_layered.argtypes = [
            wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD
        ]
        set_layered.restype = wintypes.BOOL
        COLORREF_MAGENTA = 0x00FF00FF
        LWA_COLORKEY = 0x00000001
        if not set_layered(hwnd, COLORREF_MAGENTA, 0, LWA_COLORKEY):
            raise ctypes.WinError(ctypes.get_last_error())

        # Capture exclusion is optional; it must never prevent the overlay
        # from starting or leave an opaque window covering the desktop.
        set_affinity = user32.SetWindowDisplayAffinity
        set_affinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        set_affinity.restype = wintypes.BOOL
        affinity_mode = None
        if set_affinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            affinity_mode = "EXCLUDEFROMCAPTURE"
        elif set_affinity(hwnd, 0x00000001):  # WDA_MONITOR
            affinity_mode = "MONITOR"

        self.overlay.deiconify()
        self.overlay.update_idletasks()

        set_pos = user32.SetWindowPos
        set_pos.argtypes = [
            wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, wintypes.UINT
        ]
        set_pos.restype = wintypes.BOOL
        HWND_TOPMOST = wintypes.HWND(-1)
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        SWP_SHOWWINDOW = 0x0040
        if not set_pos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW
        ):
            self.overlay.lift()

        self.overlay_hwnd = hwnd
        if affinity_mode is None:
            self.status.config(
                text="Status: AKTIF (capture exclusion unavailable)",
                fg="#65d69a"
            )

    def _change_sensor_mode(self, _selected=None):
        # Force a redraw using the selected visual style without restarting AI.
        with self.lock:
            self.patch_version += 1

    def _custom_text_changed(self, _event=None):
        with self.lock:
            self.patch_version += 1

    def _change_text_color(self, _selected=None):
        with self.lock:
            self.patch_version += 1

    def _selected_text_color(self):
        return {
            "Putih": "#ffffff",
            "Merah": "#ff4040",
            "Kuning": "#ffe45c",
            "Cyan": "#55eaff",
            "Hijau": "#65f08a",
        }.get(self.text_color.get(), "#ffffff")

    def _change_scan_target(self, _selected=None):
        # Target selection affects the next AI scan; clear old detections immediately.
        with self.lock:
            self.boxes = []
            self.previous_boxes = []
            self.track_records = []
            self.patch_data = []
            self.patch_version += 1

    @staticmethod
    def _make_static_noise(size):
        width, height = max(1, size[0]), max(1, size[1])
        noise = Image.effect_noise((width, height), 100).convert("L").convert("RGB")
        return noise

    def _render_sensor_patch(self, source_patch, mode):
        if mode in ("Blur + LOOSER", "Blur semua + fokus"):
            return source_patch.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
        if mode == "TV semut":
            return self._make_static_noise(source_patch.size)
        if mode == "Pixelate":
            w, h = source_patch.size
            tiny = source_patch.resize((max(1, w // 14), max(1, h // 14)), Image.Resampling.BILINEAR)
            return tiny.resize((w, h), Image.Resampling.NEAREST)
        if mode == "Glitch":
            image = source_patch.copy()
            draw = ImageDraw.Draw(image)
            w, h = image.size
            # Opaque dark base plus offset color bars fully conceal the target.
            draw.rectangle((0, 0, w, h), fill=(8, 10, 18))
            bar_h = max(2, h // 9)
            for i, color in enumerate(((255, 45, 90), (30, 220, 255), (245, 245, 245))):
                yy = (i * 3 + 1) * bar_h
                if yy < h:
                    draw.rectangle((0, yy, w, min(h, yy + bar_h)), fill=color)
            return image
        if mode in ("Kotak hitam", "Teks custom", "Emoji cover"):
            return Image.new("RGB", source_patch.size, (5, 5, 8))
        return source_patch.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))

    def _draw_custom_label(self, x, y, patch_w, patch_h, label, mode):
        color = self._selected_text_color()
        text = self.custom_text.get().strip() or "LOOSER"
        if mode == "Emoji cover":
            text = "😎"
            color = "#ffffff"
        if label == "face" and mode in ("LOOSER tunggal", "Teks custom", "Blur + LOOSER"):
            # Cover the eye line with a solid strip before drawing any label.
            x1 = x + int(patch_w * 0.04)
            x2 = x + int(patch_w * 0.96)
            y1 = y + int(patch_h * 0.29)
            y2 = y + int(patch_h * 0.60)
        else:
            x1, y1, x2, y2 = x, y, x + patch_w, y + patch_h
        if mode == "LOOSER tunggal":
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#050505", outline="#050505", width=0)
            display_text = text
        elif mode == "Teks custom":
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#050505", outline="#050505", width=0)
            display_text = text
        elif mode == "Blur + LOOSER":
            display_text = text
        elif mode == "Emoji cover":
            display_text = text
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#050505", outline="#050505", width=0)
        else:
            display_text = text
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#050505", outline="#050505", width=0)
        font_size = max(10, min(30, int(max(1, y2 - y1) * 0.72)))
        self.canvas.create_text(
            (x1 + x2) / 2, (y1 + y2) / 2, text=display_text,
            anchor="center", fill=color,
            font=("Segoe UI Emoji" if mode == "Emoji cover" else "Segoe UI",
                  font_size, "bold"),
            width=max(30, x2 - x1 - 6)
        )

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.status.config(text="Status: Memuat model…", fg="#ffd166")
        self.worker = threading.Thread(target=self._load_and_scan, daemon=True)
        self.worker.start()

    def _load_and_scan(self):
        try:
            self.detector = NudeDetector()
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Model deteksi wajah OpenCV tidak ditemukan.")
            if self.stop_event.is_set():
                self.root.after(0, self._stopped_ui)
                return
            self.running = True
            self.root.after(0, self._started_ui)
            self._scan_loop()
        except Exception as exc:
            self.running = False
            self.last_error = str(exc)
            self.root.after(0, lambda err=str(exc): self._failed(err))
        finally:
            self.root.after(0, self._stopped_ui)

    def _started_ui(self):
        try:
            self._configure_overlay()
            if "capture exclusion unavailable" not in self.status.cget("text"):
                self.status.config(text="Status: AKTIF", fg="#65d69a")
            self.stop_btn.config(state="normal")
        except Exception as exc:
            self.running = False
            self.stop_event.set()
            self._failed("Windows gagal mengaktifkan desktop overlay dengan aman:\n" + str(exc))

    def _screen_tiles(self, frame):
        h, w = frame.shape[:2]
        tile_w = int(w / TILE_COLS * (1 + TILE_OVERLAP))
        tile_h = int(h / TILE_ROWS * (1 + TILE_OVERLAP))
        xs = [0, max(0, w - tile_w)]
        ys = [0, max(0, h - tile_h)]
        tiles = []
        for y in ys:
            for x in xs:
                x2, y2 = min(w, x + tile_w), min(h, y + tile_h)
                tiles.append((frame[y:y2, x:x2], x, y))
        return tiles

    def _nudenet_boxes(self, frame):
        h, w = frame.shape[:2]
        found = []
        for tile, ox, oy in self._screen_tiles(frame):
            temp_name = None
            try:
                fd, temp_name = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                if not cv2.imwrite(temp_name, tile, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                    continue
                for item in self.detector.detect(temp_name):
                    label = str(item.get("class", "")).upper()
                    score = float(item.get("score", 0))
                    if label not in CENSOR_LABELS or score < LABEL_THRESHOLDS.get(label, 0.35):
                        continue
                    bx, by, bw, bh = [float(v) for v in item.get("box", [0, 0, 0, 0])]
                    px, py = int(bw * BOX_PADDING), int(bh * BOX_PADDING)
                    x1 = max(0, int(ox + bx) - px)
                    y1 = max(0, int(oy + by) - py)
                    x2 = min(w, int(ox + bx + bw) + px)
                    y2 = min(h, int(oy + by + bh) + py)
                    if x2 > x1 and y2 > y1:
                        category = "breast" if "BREAST" in label else "genital"
                        found.append((x1, y1, x2, y2, category))
            finally:
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.remove(temp_name)
                    except OSError:
                        pass
        return self._dedupe(found)

    @staticmethod
    def _dedupe(boxes):
        kept = []
        for box in sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True):
            x1,y1,x2,y2,_ = box
            cx,cy = (x1+x2)/2,(y1+y2)/2
            duplicate = False
            for old in kept:
                ox1,oy1,ox2,oy2,_ = old
                ocx,ocy = (ox1+ox2)/2,(oy1+oy2)/2
                if abs(cx-ocx) < min(x2-x1, ox2-ox1)*0.45 and abs(cy-ocy) < min(y2-y1, oy2-oy1)*0.45:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
        area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
        return intersection / max(1, area_a + area_b - intersection)

    def _smooth_boxes(self, current, previous=None, alpha=SMOOTHING_ALPHA):
        """Associate detections with short-lived motion tracks to reduce jitter."""
        records = getattr(self, "track_records", [])
        candidates = []

        # Match each new detection to the closest compatible predicted track.
        unmatched = set(range(len(records)))
        for box in current:
            x1, y1, x2, y2, label = box
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            best_index, best_cost = None, float("inf")

            for index in unmatched:
                record = records[index]
                if record["label"] != label:
                    continue
                predicted = record["box"]
                pcx = (predicted[0] + predicted[2]) / 2
                pcy = (predicted[1] + predicted[3]) / 2
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                distance = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                scale = max(bw, bh, predicted[2] - predicted[0], predicted[3] - predicted[1])
                overlap = self._box_iou(box, predicted)
                if distance < scale * 1.35 or overlap > 0.05:
                    cost = distance / max(1, scale) - overlap * 0.65
                    if cost < best_cost:
                        best_index, best_cost = index, cost

            if best_index is None:
                candidates.append({
                    "box": [x1, y1, x2, y2],
                    "label": label,
                    "velocity": [0.0, 0.0, 0.0, 0.0],
                    "misses": 0,
                })
                continue

            unmatched.remove(best_index)
            old = records[best_index]
            old_box = old["box"]
            velocity = old["velocity"]
            predicted = [old_box[i] + velocity[i] for i in range(4)]
            # Smooth against the motion prediction rather than the last raw box.
            smoothed = [
                round((1 - alpha) * predicted[i] + alpha * box[i])
                for i in range(4)
            ]
            delta = [box[i] - old_box[i] for i in range(4)]
            new_velocity = [
                0.60 * velocity[i] + 0.40 * delta[i] for i in range(4)
            ]
            candidates.append({
                "box": smoothed,
                "label": label,
                "velocity": new_velocity,
                "misses": 0,
            })

        # Keep a very short prediction through one or two missed AI scans.
        # This prevents the censor from blinking off during a brief detection miss.
        for index in unmatched:
            old = records[index]
            misses = old["misses"] + 1
            max_misses = 2 if old["label"] == "face" else 1
            if misses <= max_misses:
                velocity = old["velocity"]
                predicted = [
                    round(old["box"][i] + velocity[i] * 0.75)
                    for i in range(4)
                ]
                candidates.append({
                    "box": predicted,
                    "label": old["label"],
                    "velocity": [v * 0.65 for v in velocity],
                    "misses": misses,
                })

        self.track_records = candidates
        result = []
        for record in candidates:
            x1, y1, x2, y2 = record["box"]
            if x2 > x1 and y2 > y1:
                result.append((int(x1), int(y1), int(x2), int(y2), record["label"]))
        return result

    def _scan_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary display
            while self.running and not self.stop_event.is_set():
                started = time.perf_counter()
                raw = np.asarray(sct.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces on a downscaled frame for speed, then restore coordinates.
                scale = 0.60 if max(w, h) >= 1400 else 0.80
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                faces = self.face_cascade.detectMultiScale(
                    small, scaleFactor=1.06, minNeighbors=5, minSize=(26, 26)
                )
                boxes = []
                inv = 1.0 / scale
                selected = self.scan_target.get()
                if selected in ("Semua target", "Wajah saja"):
                    for x,y,fw,fh in faces:
                        x1,y1 = int(x*inv), int(y*inv)
                        x2,y2 = int((x+fw)*inv), int((y+fh)*inv)
                        boxes.append((x1,y1,x2,y2,"face"))
                if selected in ("Semua target", "Dada saja", "Bagian intim saja"):
                    wanted = ("breast" if selected == "Dada saja" else
                              "genital" if selected == "Bagian intim saja" else None)
                    for item_box in self._nudenet_boxes(frame):
                        if wanted is None or item_box[4] == wanted:
                            boxes.append(item_box)
                with self.lock:
                    previous = list(self.previous_boxes)
                boxes = self._smooth_boxes(boxes, previous)
                # Prepare screen patches after smoothing/prediction. The overlay
                # refreshes independently; AI detection cadence depends on the PC.
                patches = []
                screen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for x1,y1,x2,y2,label in boxes:
                    x1,y1 = max(0,x1),max(0,y1)
                    x2,y2 = min(w,x2),min(h,y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    patch = screen.crop((x1,y1,x2,y2))
                    patches.append((x1,y1,patch,label))
                full_blur = screen.filter(ImageFilter.GaussianBlur(BLUR_RADIUS * 2))
                with self.lock:
                    self.full_blur_image = full_blur
                    self.boxes = boxes
                    self.previous_boxes = boxes
                    self.patch_data = patches
                    self.patch_version += 1
                    self.frame_size = (w,h)
                elapsed = time.perf_counter() - started
                time.sleep(max(0.02, DETECTION_INTERVAL - elapsed))

    def refresh_overlay(self):
        if self.running and self.overlay.state() != "withdrawn":
            # Poll at 60 Hz, but rebuild canvas items only when detections change.
            with self.lock:
                version = self.patch_version
                patches = list(self.patch_data) if version != self.rendered_patch_version else None
                mode = self.sensor_mode.get()

            if patches is not None:
                self.canvas.delete("all")
                self.patch_images = []
                if mode == "Blur semua + fokus":
                    with self.lock:
                        background = self.full_blur_image.copy() if self.full_blur_image else None
                    if background is not None:
                        # Magenta cut-outs reveal the original desktop only inside detected target boxes.
                        draw = ImageDraw.Draw(background)
                        for x, y, source_patch, label in patches:
                            draw.rectangle((x, y, x + source_patch.width, y + source_patch.height),
                                           fill=(255, 0, 255))
                        photo = ImageTk.PhotoImage(background)
                        self.patch_images.append(photo)
                        self.canvas.create_image(0, 0, image=photo, anchor="nw")
                else:
                    for x, y, source_patch, label in patches:
                        patch = self._render_sensor_patch(source_patch, mode)
                        photo = ImageTk.PhotoImage(patch)
                        self.patch_images.append(photo)
                        self.canvas.create_image(x, y, image=photo, anchor="nw")
                        if mode in ("LOOSER tunggal", "Teks custom", "Emoji cover", "Blur + LOOSER"):
                            patch_w, patch_h = source_patch.size
                            self._draw_custom_label(x, y, patch_w, patch_h, label, mode)
                self.rendered_patch_version = version
            self.overlay.lift()
        self.root.after(OVERLAY_REFRESH_MS, self.refresh_overlay)

    def _failed(self, error):
        self.status.config(text="Sensor gagal / berhenti", fg="#ff7777")
        messagebox.showerror(APP_TITLE, "Gagal menjalankan sensor:\n\n" + error)

    def _stopped_ui(self):
        self.running = False
        try:
            self.overlay.withdraw()
        except Exception:
            pass
        with self.lock:
            self.boxes = []
            self.previous_boxes = []
            self.track_records = []
            self.patch_data = []
        self.canvas.delete("all")
        self.patch_images = []
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if not self.closing:
            self.status.config(text="Status: Mati", fg="#ff7777")

    def stop(self):
        self.running = False
        self.stop_event.set()
        self._stopped_ui()

    def close(self):
        self.closing = True
        self.stop()
        if self.hotkey_registered:
            try:
                ctypes.windll.user32.UnregisterHotKey(None, 0x5343)
            except Exception:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenCensor(root)
    root.mainloop()
