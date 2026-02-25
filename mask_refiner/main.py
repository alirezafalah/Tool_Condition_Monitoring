"""
Interactive Mask Refiner (OpenGL-accelerated)

Purpose:
    Refine existing tool masks against blurred input frames.

Dataset layout (auto-detected):
    DATA/blurred/{tool_id}_blurred/
    DATA/masks/{tool_id}_final_masks/

Features:
    - Smart Select (GrabCut)
    - Magic Wand
    - Brush / Eraser (circle + square)
    - Lasso / Polygon / Rectangle / Contour Snap
    - Add/Subtract mode
    - Undo / Redo (Ctrl+Z / Ctrl+Y)
    - Zoom (wheel) / Pan (middle drag)
    - Auto-save on image navigation

Usage:
    python -m mask_refiner.main
    python -m mask_refiner.main --tool tool002
    python -m mask_refiner.main --data-dir ".../DATA"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QShortcut,
    QSurfaceFormat,
    QWheelEvent,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glViewport,
)


IMAGE_EXTS = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")

# --- Performance: use half of CPU cores (capped at 8) for OpenCV internal threading,
#     and a small thread pool to offload heavy CV ops from the GUI thread. ---
_NUM_CV_THREADS = min(8, max(4, (os.cpu_count() or 8) // 2))
_worker_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv_worker")


def _configure_cv_performance():
    """Set up OpenCV multi-threading and GPU (OpenCL) acceleration."""
    cv2.setNumThreads(_NUM_CV_THREADS)
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass
    # Report configuration
    ocl_status = "enabled" if cv2.ocl.useOpenCL() else "disabled"
    print(f"[perf] OpenCV threads: {_NUM_CV_THREADS}, OpenCL: {ocl_status}")
    try:
        dev = cv2.ocl.Device.getDefault()
        if dev and dev.name():
            print(f"[perf] OpenCL device: {dev.name()}")
    except Exception:
        pass


def build_composite_rgba(cv_image: np.ndarray, mask: np.ndarray, mask_blend: float = 1.0) -> np.ndarray:
    """Build RGBA composite with adjustable mask blending.
    
    Args:
        cv_image: Input BGR image
        mask: Binary mask (0 = background, >0 = foreground)
        mask_blend: Blending amount (0.0 = hidden, 1.0 = full visibility)
    """
    h, w = cv_image.shape[:2]
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    selected = mask > 0
    # Apply mask coloring with adjustable blend
    overlay = rgb.copy()
    overlay[selected] = overlay[selected] * 0.5 + np.array([0, 180, 0], dtype=np.float32) * 0.5
    overlay[~selected] = overlay[~selected] * 0.7
    
    # Blend between original image and overlay based on mask_blend
    rgb = rgb * (1.0 - mask_blend) + overlay * mask_blend

    # Draw contours only if mask_blend > 0.1
    if mask_blend > 0.1:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp_bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.drawContours(temp_bgr, contours, -1, (255, 255, 0), max(1, w // 800))
        rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255)
    rgba[:, :, 3] = 255
    return rgba


def _sort_key(path: str) -> Tuple[float, str]:
    stem = Path(path).stem
    matches = re.findall(r"\d+(?:\.\d+)?", stem)
    if matches:
        try:
            return float(matches[-1]), stem.lower()
        except ValueError:
            pass
    return float("inf"), stem.lower()


def discover_tool_ids(blurred_root: str, masks_root: str) -> List[str]:
    if not os.path.isdir(blurred_root):
        return []

    blurred_ids = set()
    for name in os.listdir(blurred_root):
        m = re.match(r"(tool\d+)_blurred(?:_original)?$", name)
        if m and os.path.isdir(os.path.join(blurred_root, name)):
            blurred_ids.add(m.group(1))

    mask_ids = set()
    if os.path.isdir(masks_root):
        for name in os.listdir(masks_root):
            m = re.match(r"(tool\d+).*_final_masks(?:_original)?$", name)
            if m and os.path.isdir(os.path.join(masks_root, name)):
                mask_ids.add(m.group(1))

    if mask_ids:
        return sorted(blurred_ids & mask_ids) or sorted(blurred_ids)
    return sorted(blurred_ids)


def resolve_blurred_folder(tool_id: str, blurred_root: str) -> Optional[str]:
    candidates = [
        os.path.join(blurred_root, f"{tool_id}_blurred"),
        os.path.join(blurred_root, f"{tool_id}_blurred_original"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path

    if os.path.isdir(blurred_root):
        for name in sorted(os.listdir(blurred_root)):
            if name.startswith(tool_id) and "blurred" in name:
                path = os.path.join(blurred_root, name)
                if os.path.isdir(path):
                    return path
    return None


def resolve_masks_folder(tool_id: str, masks_root: str) -> str:
    candidates = [
        os.path.join(masks_root, f"{tool_id}_final_masks"),
        os.path.join(masks_root, f"{tool_id}gain10paperBG_final_masks"),
        os.path.join(masks_root, f"{tool_id}gain10_final_masks"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path

    if os.path.isdir(masks_root):
        for name in sorted(os.listdir(masks_root)):
            if name.startswith(tool_id) and name.endswith("_final_masks"):
                path = os.path.join(masks_root, name)
                if os.path.isdir(path):
                    return path

    os.makedirs(masks_root, exist_ok=True)
    fallback = candidates[0]
    os.makedirs(fallback, exist_ok=True)
    return fallback


class GLCanvas(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        super().__init__(parent)
        self.setFormat(fmt)
        self.main_window = parent
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.cv_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self._img_w = 0
        self._img_h = 0

        self._texture_id = 0
        self._texture_dirty = True

        self.zoom_level = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._pan_active = False
        self._pan_start = QPoint(0, 0)

        self._drawing = False
        self._points: List[QPoint] = []
        self._rect_start: Optional[QPoint] = None
        self._rect_end: Optional[QPoint] = None
        self._brush_last: Optional[QPoint] = None

        self._cursor_pos: Optional[QPoint] = None

        # GrabCut async state
        self._grabcut_future: Optional[Future] = None
        self._grabcut_timer: Optional[QTimer] = None

    def mark_texture_dirty(self):
        self._texture_dirty = True
        self.update()

    def _upload_texture(self):
        if self.cv_image is None or self.mask is None:
            return

        rgba = np.ascontiguousarray(build_composite_rgba(self.cv_image, self.mask, self.main_window.mask_blend if self.main_window else 1.0))
        h, w = rgba.shape[:2]
        self._img_w, self._img_h = w, h

        if self._texture_id == 0:
            self._texture_id = int(glGenTextures(1))

        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._texture_dirty = False

    def _origin(self) -> Tuple[float, float]:
        dw = self._img_w * self.zoom_level
        dh = self._img_h * self.zoom_level
        ox = (self.width() - dw) / 2.0 + self.pan_offset.x()
        oy = (self.height() - dh) / 2.0 + self.pan_offset.y()
        return ox, oy

    def image_pos(self, widget_pos: QPoint) -> Optional[QPoint]:
        if self.cv_image is None:
            return None
        ox, oy = self._origin()
        ix = int((widget_pos.x() - ox) / self.zoom_level)
        iy = int((widget_pos.y() - oy) / self.zoom_level)
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            return QPoint(ix, iy)
        return None

    def widget_from_image(self, ix: int, iy: int) -> QPoint:
        ox, oy = self._origin()
        return QPoint(int(ix * self.zoom_level + ox), int(iy * self.zoom_level + oy))

    def initializeGL(self):
        glClearColor(0.16, 0.16, 0.16, 1.0)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.cv_image is None:
            return

        if self._texture_dirty or self._texture_id == 0:
            self._upload_texture()

        if self._texture_id == 0:
            return

        ww, wh = self.width(), self.height()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, ww, wh, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        ox, oy = self._origin()
        dw = self._img_w * self.zoom_level
        dh = self._img_h * self.zoom_level

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(ox, oy)
        glTexCoord2f(1, 0)
        glVertex2f(ox + dw, oy)
        glTexCoord2f(1, 1)
        glVertex2f(ox + dw, oy + dh)
        glTexCoord2f(0, 1)
        glVertex2f(ox, oy + dh)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_overlays(painter)
        painter.end()

    def _draw_overlays(self, painter: QPainter):
        mw = self.main_window
        if mw is None:
            return
        tool = mw.current_tool

        if tool in ("rectangle", "smart_select") and self._rect_start and self._rect_end:
            color = QColor(0, 200, 255) if tool == "smart_select" else QColor(255, 255, 0)
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            p1 = self.widget_from_image(self._rect_start.x(), self._rect_start.y())
            p2 = self.widget_from_image(self._rect_end.x(), self._rect_end.y())
            painter.drawRect(QRect(p1, p2))

        if tool == "lasso" and len(self._points) > 1:
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            for i in range(1, len(self._points)):
                p1 = self.widget_from_image(self._points[i - 1].x(), self._points[i - 1].y())
                p2 = self.widget_from_image(self._points[i].x(), self._points[i].y())
                painter.drawLine(p1, p2)

        if tool == "polygon" and self._points:
            painter.setPen(QPen(QColor(255, 100, 0), 2))
            for i, pt in enumerate(self._points):
                wp = self.widget_from_image(pt.x(), pt.y())
                painter.drawEllipse(wp, 5, 5)
                if i > 0:
                    prev = self.widget_from_image(self._points[i - 1].x(), self._points[i - 1].y())
                    painter.drawLine(prev, wp)
            if self._cursor_pos:
                painter.setPen(QPen(QColor(255, 100, 0, 120), 1, Qt.PenStyle.DashLine))
                last = self.widget_from_image(self._points[-1].x(), self._points[-1].y())
                painter.drawLine(last, self._cursor_pos)

        if tool in ("brush", "eraser") and self._cursor_pos:
            radius = max(1, int(mw.brush_size * self.zoom_level))
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if mw.brush_shape == "square":
                size = radius * 2
                painter.drawRect(self._cursor_pos.x() - radius, self._cursor_pos.y() - radius, size, size)
            else:
                painter.drawEllipse(self._cursor_pos, radius, radius)

    def wheelEvent(self, event: QWheelEvent):
        old_zoom = self.zoom_level
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = max(0.02, min(old_zoom * factor, 50.0))

        mouse = event.position()
        ox, oy = self._origin()
        img_x = (mouse.x() - ox) / old_zoom
        img_y = (mouse.y() - oy) / old_zoom

        self.zoom_level = new_zoom
        new_ox = mouse.x() - img_x * new_zoom
        new_oy = mouse.y() - img_y * new_zoom
        target_ox = (self.width() - self._img_w * new_zoom) / 2.0
        target_oy = (self.height() - self._img_h * new_zoom) / 2.0
        self.pan_offset = QPointF(new_ox - target_ox, new_oy - target_oy)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.cv_image is None or self.main_window is None:
            return
        img_pt = self.image_pos(event.pos())
        if img_pt is None:
            return

        tool = self.main_window.current_tool
        if event.button() != Qt.MouseButton.LeftButton:
            return

        self._drawing = True

        if tool in ("brush", "eraser"):
            self._push_undo()
            self._brush_last = img_pt
            self._apply_brush(img_pt, tool)
            self.mark_texture_dirty()
        elif tool in ("rectangle", "smart_select"):
            self._rect_start = img_pt
            self._rect_end = img_pt
            self.update()
        elif tool == "lasso":
            self._points = [img_pt]
        elif tool == "polygon":
            if len(self._points) >= 3:
                first = self._points[0]
                dist = ((img_pt.x() - first.x()) ** 2 + (img_pt.y() - first.y()) ** 2) ** 0.5
                if dist < 15:
                    self._finish_polygon()
                    self.mark_texture_dirty()
                    return
            self._points.append(img_pt)
            self.update()
        elif tool == "magic_wand":
            self._apply_magic_wand(img_pt)
            self.mark_texture_dirty()
        elif tool == "contour_snap":
            self._apply_contour_snap(img_pt)
            self.mark_texture_dirty()

    def mouseMoveEvent(self, event: QMouseEvent):
        self._cursor_pos = event.pos()

        if self._pan_active:
            delta = event.pos() - self._pan_start
            self.pan_offset = QPointF(self.pan_offset.x() + delta.x(), self.pan_offset.y() + delta.y())
            self._pan_start = event.pos()
            self.update()
            return

        if not self._drawing:
            if self.main_window and self.main_window.current_tool in ("brush", "eraser", "polygon"):
                self.update()
            return

        img_pt = self.image_pos(event.pos())
        if img_pt is None or self.main_window is None:
            return

        tool = self.main_window.current_tool
        if tool in ("brush", "eraser"):
            self._apply_brush_line(self._brush_last, img_pt, tool)
            self._brush_last = img_pt
            self.mark_texture_dirty()
        elif tool in ("rectangle", "smart_select"):
            self._rect_end = img_pt
            self.update()
        elif tool == "lasso":
            self._points.append(img_pt)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if not self._drawing:
            return

        self._drawing = False
        tool = self.main_window.current_tool if self.main_window else ""

        if tool == "rectangle" and self._rect_start and self._rect_end:
            self._finish_rectangle()
        elif tool == "smart_select" and self._rect_start and self._rect_end:
            self._finish_grabcut()
        elif tool == "lasso" and len(self._points) > 2:
            self._finish_lasso()
        elif tool in ("brush", "eraser"):
            self._brush_last = None

        self.mark_texture_dirty()
        if self.main_window:
            self.main_window._update_mask_info()

    def _push_undo(self):
        if self.main_window:
            self.main_window.push_undo()

    def _is_subtract(self) -> bool:
        return self.main_window is not None and self.main_window.current_mode == "subtract"

    def _apply_to_mask(self, tool_mask: np.ndarray):
        if self.mask is None:
            return
        self._push_undo()
        if self._is_subtract():
            self.mask[tool_mask > 0] = 0
        else:
            self.mask[tool_mask > 0] = 255

    def _apply_brush(self, pt: QPoint, tool: str):
        if self.mask is None or self.main_window is None:
            return
        r = self.main_window.brush_size
        shape = self.main_window.brush_shape
        val = 0 if (tool == "eraser" or self._is_subtract()) else 255
        if shape == "square":
            cv2.rectangle(self.mask, (pt.x() - r, pt.y() - r), (pt.x() + r, pt.y() + r), int(val), -1)
        else:
            cv2.circle(self.mask, (pt.x(), pt.y()), r, int(val), -1)

    def _apply_brush_line(self, pt1: Optional[QPoint], pt2: QPoint, tool: str):
        if self.mask is None or pt1 is None or self.main_window is None:
            return
        r = self.main_window.brush_size
        shape = self.main_window.brush_shape
        val = 0 if (tool == "eraser" or self._is_subtract()) else 255
        if shape == "square":
            cv2.line(self.mask, (pt1.x(), pt1.y()), (pt2.x(), pt2.y()), int(val), r * 2, lineType=cv2.LINE_8)
        else:
            cv2.line(self.mask, (pt1.x(), pt1.y()), (pt2.x(), pt2.y()), int(val), r * 2)

    def _finish_rectangle(self):
        if self.mask is None:
            return
        x1 = min(self._rect_start.x(), self._rect_end.x())
        y1 = min(self._rect_start.y(), self._rect_end.y())
        x2 = max(self._rect_start.x(), self._rect_end.x())
        y2 = max(self._rect_start.y(), self._rect_end.y())
        tool_mask = np.zeros_like(self.mask)
        cv2.rectangle(tool_mask, (x1, y1), (x2, y2), 255, -1)
        self._apply_to_mask(tool_mask)
        self._rect_start = self._rect_end = None

    def _finish_lasso(self):
        if self.mask is None:
            return
        pts = np.array([(p.x(), p.y()) for p in self._points], dtype=np.int32)
        tool_mask = np.zeros_like(self.mask)
        cv2.fillPoly(tool_mask, [pts], 255)
        self._apply_to_mask(tool_mask)
        self._points = []

    def _finish_polygon(self):
        if self.mask is None:
            return
        pts = np.array([(p.x(), p.y()) for p in self._points], dtype=np.int32)
        tool_mask = np.zeros_like(self.mask)
        cv2.fillPoly(tool_mask, [pts], 255)
        self._apply_to_mask(tool_mask)
        self._points = []

    def _finish_grabcut(self):
        if self.cv_image is None or self.mask is None or self.main_window is None:
            return

        x1 = min(self._rect_start.x(), self._rect_end.x())
        y1 = min(self._rect_start.y(), self._rect_end.y())
        x2 = max(self._rect_start.x(), self._rect_end.x())
        y2 = max(self._rect_start.y(), self._rect_end.y())
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            self._rect_start = self._rect_end = None
            return

        self.main_window.statusBar().showMessage("Running GrabCut... please wait")
        self._rect_start = self._rect_end = None

        # Prepare data copies so the worker thread doesn't touch GUI-owned arrays
        image_copy = self.cv_image.copy()
        mask_copy = self.mask.copy()
        rect = (x1, y1, w, h)

        def _grabcut_worker() -> Optional[np.ndarray]:
            gc_mask = np.zeros(image_copy.shape[:2], dtype=np.uint8)
            if np.any(mask_copy > 0):
                gc_mask[mask_copy > 0] = cv2.GC_PR_FGD
                gc_mask[mask_copy == 0] = cv2.GC_PR_BGD
                outside = np.ones_like(gc_mask, dtype=bool)
                outside[y1:y1 + h, x1:x1 + w] = False
                gc_mask[outside] = cv2.GC_BGD
                mode = cv2.GC_INIT_WITH_MASK
            else:
                mode = cv2.GC_INIT_WITH_RECT
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(image_copy, gc_mask, rect, bgd, fgd, 5, mode)
                return np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except cv2.error:
                return None

        self._grabcut_future = _worker_pool.submit(_grabcut_worker)

        # Poll for completion without blocking the GUI
        self._grabcut_timer = QTimer()
        self._grabcut_timer.setInterval(40)
        self._grabcut_timer.timeout.connect(self._poll_grabcut)
        self._grabcut_timer.start()

    def _poll_grabcut(self):
        """Check if the background GrabCut has finished."""
        if self._grabcut_future is None or not self._grabcut_future.done():
            return
        self._grabcut_timer.stop()
        self._grabcut_timer = None

        result = self._grabcut_future.result()
        self._grabcut_future = None

        if result is not None:
            self._apply_to_mask(result)
            if self.main_window:
                self.main_window.statusBar().showMessage("GrabCut done", 3000)
        else:
            if self.main_window:
                self.main_window.statusBar().showMessage("GrabCut failed", 5000)

        self.mark_texture_dirty()
        if self.main_window:
            self.main_window._update_mask_info()

    def _apply_magic_wand(self, pt: QPoint):
        if self.cv_image is None or self.mask is None or self.main_window is None:
            return

        tol = self.main_window.wand_tolerance
        h, w = self.cv_image.shape[:2]
        lab = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2LAB)
        seed = lab[pt.y(), pt.x()].astype(np.int16)
        diff = np.abs(lab.astype(np.int16) - seed)
        within = (np.max(diff, axis=2) <= tol).astype(np.uint8) * 255

        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(within, flood_mask, (pt.x(), pt.y()), 128, loDiff=0, upDiff=0, flags=4 | (128 << 8))
        tool_mask = (within == 128).astype(np.uint8) * 255
        self._apply_to_mask(tool_mask)

    def _apply_contour_snap(self, pt: QPoint):
        if self.cv_image is None or self.mask is None:
            return

        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        best = None
        best_dist = float("inf")
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            dist = abs(cv2.pointPolygonTest(cnt, (float(pt.x()), float(pt.y())), True))
            if dist < best_dist:
                best_dist = dist
                best = cnt

        if best is None or best_dist > 100:
            return

        tool_mask = np.zeros_like(self.mask)
        cv2.drawContours(tool_mask, [best], -1, 255, -1)
        self._apply_to_mask(tool_mask)

    def fit_image(self):
        if self.cv_image is None:
            return
        self._img_h, self._img_w = self.cv_image.shape[:2]
        self.zoom_level = min(self.width() / self._img_w, self.height() / self._img_h) * 0.95
        self.pan_offset = QPointF(0.0, 0.0)
        self.mark_texture_dirty()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def closeEvent(self, event):
        if self._texture_id:
            try:
                glDeleteTextures([self._texture_id])
            except Exception:
                pass
            self._texture_id = 0
        super().closeEvent(event)


class MaskRefinerWindow(QMainWindow):
    def __init__(self, data_dir: str, initial_tool: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("PixelGrid — Tool Mask Refiner (OpenGL)")
        self.setMinimumSize(1150, 760)
        self.resize(1450, 920)

        self.data_dir = os.path.abspath(data_dir)
        self.blurred_root = os.path.join(self.data_dir, "blurred")
        self.masks_root = os.path.join(self.data_dir, "masks")

        self.tool_ids = discover_tool_ids(self.blurred_root, self.masks_root)
        self.current_tool_id = ""
        self.frames_folder = ""
        self.masks_folder = ""
        self._manual_mode = False

        self.image_paths: List[str] = []
        self.mask_path_for_image: Dict[str, str] = {}
        self.current_index = -1

        self.current_tool = "brush"
        self.current_mode = "add"
        self.brush_size = 20
        self.brush_shape = "circle"
        self.wand_tolerance = 25
        self.mask_blend = 1.0  # Mask visibility (0.0 = hidden, 1.0 = full)

        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self._max_undo = 50
        self._dirty = False

        self._build_ui()

        if initial_tool and initial_tool in self.tool_ids:
            self._tool_selector.setCurrentText(initial_tool)
        elif self.tool_ids:
            self._tool_selector.setCurrentIndex(0)
        else:
            self.statusBar().showMessage("No tool folders detected under DATA/blurred")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        left = QWidget()
        left.setFixedWidth(245)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        tg = QGroupBox("Selection Tools")
        tl = QVBoxLayout(tg)
        tl.setSpacing(2)
        self._tool_buttons: Dict[str, QPushButton] = {}
        for tid, label in [
            ("smart_select", "🎯 Smart Select (GrabCut)"),
            ("magic_wand", "🪄 Magic Wand"),
            ("brush", "🖌 Brush"),
            ("eraser", "🧹 Eraser"),
            ("lasso", "✏️ Lasso (Freeform)"),
            ("polygon", "⬡ Polygon"),
            ("rectangle", "▭ Rectangle"),
            ("contour_snap", "🔲 Contour Snap"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet("text-align: left; padding: 5px 10px;")
            btn.clicked.connect(lambda _, t=tid: self._select_tool(t))
            tl.addWidget(btn)
            self._tool_buttons[tid] = btn
        ll.addWidget(tg)

        mg = QGroupBox("Mode")
        ml = QHBoxLayout(mg)
        self._add_radio = QRadioButton("➕ Add")
        self._sub_radio = QRadioButton("➖ Subtract")
        self._add_radio.setChecked(True)
        self._add_radio.toggled.connect(lambda c: self._set_mode("add") if c else None)
        self._sub_radio.toggled.connect(lambda c: self._set_mode("subtract") if c else None)
        ml.addWidget(self._add_radio)
        ml.addWidget(self._sub_radio)
        ll.addWidget(mg)

        pg = QGroupBox("Parameters")
        pl = QVBoxLayout(pg)

        pl.addWidget(QLabel("Brush Size:"))
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 150)
        self._brush_slider.setValue(self.brush_size)
        self._brush_label = QLabel(str(self.brush_size))
        self._brush_slider.valueChanged.connect(self._on_brush_size)
        r1 = QHBoxLayout()
        r1.addWidget(self._brush_slider)
        r1.addWidget(self._brush_label)
        pl.addLayout(r1)

        pl.addWidget(QLabel("Brush Shape:"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(["Circle", "Square"])
        self._shape_combo.currentTextChanged.connect(self._on_brush_shape)
        pl.addWidget(self._shape_combo)

        pl.addWidget(QLabel("Wand Tolerance:"))
        self._tol_slider = QSlider(Qt.Orientation.Horizontal)
        self._tol_slider.setRange(1, 120)
        self._tol_slider.setValue(self.wand_tolerance)
        self._tol_label = QLabel(str(self.wand_tolerance))
        self._tol_slider.valueChanged.connect(self._on_tolerance)
        r2 = QHBoxLayout()
        r2.addWidget(self._tol_slider)
        r2.addWidget(self._tol_label)
        pl.addLayout(r2)

        pl.addWidget(QLabel("Mask Visibility:"))
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(int(self.mask_blend * 100))
        self._blend_label = QLabel("100%")
        self._blend_slider.valueChanged.connect(self._on_blend_changed)
        r3 = QHBoxLayout()
        r3.addWidget(self._blend_slider)
        r3.addWidget(self._blend_label)
        pl.addLayout(r3)

        ll.addWidget(pg)

        ag = QGroupBox("Actions")
        al = QVBoxLayout(ag)
        for label, fn in [
            ("🗑 Clear Mask", self._clear_mask),
            ("🔄 Invert Mask", self._invert_mask),
            ("⊕ Dilate (grow)", self._dilate_mask),
            ("⊖ Erode (shrink)", self._erode_mask),
            ("〰 Smooth Edges", self._smooth_mask),
            ("⬛ Fill Holes", self._fill_holes),
        ]:
            b = QPushButton(label)
            b.clicked.connect(fn)
            al.addWidget(b)
        ll.addWidget(ag)
        ll.addStretch()

        self.canvas = GLCanvas(self)

        right = QWidget()
        right.setFixedWidth(260)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        dg = QGroupBox("Dataset")
        dl = QVBoxLayout(dg)
        dl.addWidget(QLabel("Tool:"))
        self._tool_selector = QComboBox()
        self._tool_selector.addItems(self.tool_ids)
        self._tool_selector.currentTextChanged.connect(self._on_tool_changed)
        dl.addWidget(self._tool_selector)
        pick_frames_btn = QPushButton("📂 Select Blurred Folder...")
        pick_frames_btn.clicked.connect(self._select_frames_folder)
        dl.addWidget(pick_frames_btn)
        pick_masks_btn = QPushButton("📂 Select Masks Folder...")
        pick_masks_btn.clicked.connect(self._select_masks_folder)
        dl.addWidget(pick_masks_btn)
        reload_btn = QPushButton("🔄 Reload Current Folders")
        reload_btn.clicked.connect(self._reload_current_paths)
        dl.addWidget(reload_btn)
        self._paths_label = QLabel("")
        self._paths_label.setWordWrap(True)
        self._paths_label.setStyleSheet("color: #aaa; font-size: 10px;")
        dl.addWidget(self._paths_label)
        rl.addWidget(dg)

        ng = QGroupBox("Navigation")
        nl = QVBoxLayout(ng)
        self._img_label = QLabel("No images")
        self._img_label.setWordWrap(True)
        self._img_label.setStyleSheet("font-weight: bold;")
        self._counter_label = QLabel("")
        nl.addWidget(self._img_label)
        nl.addWidget(self._counter_label)
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        nl.addLayout(nav_row)
        b_save = QPushButton("💾 Save Mask (Ctrl+S)")
        b_save.clicked.connect(self._save_mask)
        nl.addWidget(b_save)
        b_fit = QPushButton("🔍 Fit View")
        b_fit.clicked.connect(self.canvas.fit_image)
        nl.addWidget(b_fit)
        rl.addWidget(ng)

        ug = QGroupBox("History")
        ul = QVBoxLayout(ug)
        b_u = QPushButton("↩ Undo (Ctrl+Z)")
        b_u.clicked.connect(self._undo)
        ul.addWidget(b_u)
        b_r = QPushButton("↪ Redo (Ctrl+Y)")
        b_r.clicked.connect(self._redo)
        ul.addWidget(b_r)
        rl.addWidget(ug)

        vg = QGroupBox("View")
        vl = QVBoxLayout(vg)
        self._show_mask_edges = QCheckBox("Show mask edges")
        self._show_mask_edges.setChecked(True)
        self._show_mask_edges.toggled.connect(lambda _: self.canvas.mark_texture_dirty())
        vl.addWidget(self._show_mask_edges)
        rl.addWidget(vg)

        ig = QGroupBox("Mask Info")
        il = QVBoxLayout(ig)
        self._mask_info_label = QLabel("No mask")
        self._mask_info_label.setWordWrap(True)
        il.addWidget(self._mask_info_label)
        rl.addWidget(ig)
        rl.addStretch()

        sg = QGroupBox("Shortcuts")
        sl = QVBoxLayout(sg)
        st = QLabel("\n".join([
            "B — Brush", "E — Eraser", "W — Magic Wand",
            "G — GrabCut", "L — Lasso", "P — Polygon",
            "R — Rectangle", "C — Contour Snap",
            "A — Add mode", "S — Subtract mode",
            "[ / ] — Brush size", "Scroll — Zoom",
            "Mid-click — Pan", "Ctrl+Z / Ctrl+Y",
            "← / → — Prev / Next", "Ctrl+S — Save", "F — Fit",
        ]))
        st.setStyleSheet("font-size: 10px; color: #bbb;")
        sl.addWidget(st)
        rl.addWidget(sg)

        main_layout.addWidget(left)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(right)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self._setup_shortcuts()
        self._select_tool("brush")

    def _setup_shortcuts(self):
        for key, fn in {
            "B": lambda: self._select_tool("brush"),
            "E": lambda: self._select_tool("eraser"),
            "W": lambda: self._select_tool("magic_wand"),
            "G": lambda: self._select_tool("smart_select"),
            "L": lambda: self._select_tool("lasso"),
            "P": lambda: self._select_tool("polygon"),
            "R": lambda: self._select_tool("rectangle"),
            "C": lambda: self._select_tool("contour_snap"),
            "A": lambda: self._set_mode("add"),
            "S": lambda: self._set_mode("subtract"),
            "F": self.canvas.fit_image,
        }.items():
            QShortcut(QKeySequence(key), self).activated.connect(fn)

        QShortcut(QKeySequence("["), self).activated.connect(
            lambda: self._brush_slider.setValue(max(1, self.brush_size - 5))
        )
        QShortcut(QKeySequence("]"), self).activated.connect(
            lambda: self._brush_slider.setValue(min(150, self.brush_size + 5))
        )
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self).activated.connect(self._redo)
        QShortcut(QKeySequence.StandardKey.Save, self).activated.connect(self._save_mask)
        QShortcut(QKeySequence("Left"), self).activated.connect(self._prev_image)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._next_image)

    def _select_tool(self, tid: str):
        self.current_tool = tid
        for tool_name, btn in self._tool_buttons.items():
            btn.setChecked(tool_name == tid)
        if tid != "polygon":
            self.canvas._points = []
        self.statusBar().showMessage(f"Tool: {tid.replace('_', ' ').title()}")
        self.canvas.update()

    def _set_mode(self, mode: str):
        self.current_mode = mode
        (self._add_radio if mode == "add" else self._sub_radio).setChecked(True)
        self.statusBar().showMessage(f"Mode: {mode.title()}")

    def _on_brush_size(self, value: int):
        self.brush_size = value
        self._brush_label.setText(str(value))

    def _on_brush_shape(self, value: str):
        self.brush_shape = value.strip().lower()

    def _on_tolerance(self, value: int):
        self.wand_tolerance = value
        self._tol_label.setText(str(value))

    def _on_blend_changed(self, value: int):
        self.mask_blend = value / 100.0
        self._blend_label.setText(f"{value}%")
        self.canvas.mark_texture_dirty()

    def _refresh_paths_label(self):
        mode = "Manual" if self._manual_mode else "Auto"
        self._paths_label.setText(
            f"Mode: {mode}\nFrames: {self.frames_folder or '-'}\nMasks: {self.masks_folder or '-'}"
        )

    def _reload_current_paths(self):
        if not self.frames_folder or not self.masks_folder:
            self.statusBar().showMessage("Select tool or folders first", 3000)
            return
        self._apply_dataset_paths(self.frames_folder, self.masks_folder, auto_save=True)

    def _select_frames_folder(self):
        start_dir = self.frames_folder or self.blurred_root
        selected = QFileDialog.getExistingDirectory(self, "Select Blurred Frames Folder", start_dir)
        if not selected:
            return

        self._manual_mode = True
        self.frames_folder = selected

        if not self.masks_folder:
            name = os.path.basename(selected.rstrip("/\\"))
            m = re.match(r"(tool\d+)_blurred(?:_original)?$", name)
            if m:
                self.masks_folder = resolve_masks_folder(m.group(1), self.masks_root)

        if self.masks_folder:
            self._apply_dataset_paths(self.frames_folder, self.masks_folder, auto_save=True)
        else:
            self._refresh_paths_label()
            self.statusBar().showMessage("Now select a masks folder", 3000)

    def _select_masks_folder(self):
        start_dir = self.masks_folder or self.masks_root
        selected = QFileDialog.getExistingDirectory(self, "Select Masks Folder", start_dir)
        if not selected:
            return

        self._manual_mode = True
        self.masks_folder = selected

        if self.frames_folder:
            self._apply_dataset_paths(self.frames_folder, self.masks_folder, auto_save=True)
        else:
            self._refresh_paths_label()
            self.statusBar().showMessage("Now select a blurred frames folder", 3000)

    def _apply_dataset_paths(self, frames_folder: str, masks_folder: str, auto_save: bool = True):
        if auto_save and self.current_index >= 0 and self.canvas.mask is not None and self._dirty:
            self._save_mask(silent=True)

        if not os.path.isdir(frames_folder):
            self.statusBar().showMessage(f"Frames folder not found: {frames_folder}", 5000)
            return

        os.makedirs(masks_folder, exist_ok=True)
        self.frames_folder = frames_folder
        self.masks_folder = masks_folder
        self._refresh_paths_label()

        self._load_image_list()
        if self.image_paths:
            self._go_to_image(0, auto_save=False)
        else:
            self.current_index = -1
            self.canvas.cv_image = None
            self.canvas.mask = None
            self._img_label.setText("No images")
            self._counter_label.setText("")
            self._update_mask_info()
            self.canvas.mark_texture_dirty()
            self.statusBar().showMessage("No images found in selected frames folder", 4000)

    def _on_tool_changed(self, tool_id: str):
        if not tool_id:
            return

        self.current_tool_id = tool_id
        self._manual_mode = False
        blurred = resolve_blurred_folder(tool_id, self.blurred_root)
        if not blurred:
            self.statusBar().showMessage(f"Missing blurred folder for {tool_id}", 5000)
            return

        masks = resolve_masks_folder(tool_id, self.masks_root)
        self._apply_dataset_paths(blurred, masks, auto_save=True)

    def _load_image_list(self):
        if not self.frames_folder or not os.path.isdir(self.frames_folder):
            self.image_paths = []
            self.mask_path_for_image.clear()
            return

        paths = []
        for ext in IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(self.frames_folder, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(self.frames_folder, f"*{ext.upper()}")))
        self.image_paths = sorted(set(paths), key=_sort_key)

        self.mask_path_for_image.clear()
        if not os.path.isdir(self.masks_folder):
            os.makedirs(self.masks_folder, exist_ok=True)

        existing_masks = []
        for ext in IMAGE_EXTS:
            existing_masks.extend(glob.glob(os.path.join(self.masks_folder, f"*{ext}")))
            existing_masks.extend(glob.glob(os.path.join(self.masks_folder, f"*{ext.upper()}")))

        by_name = {os.path.basename(p).lower(): p for p in existing_masks}
        by_stem: Dict[str, List[str]] = {}
        for path in existing_masks:
            by_stem.setdefault(Path(path).stem.lower(), []).append(path)

        for frame_path in self.image_paths:
            fname = os.path.basename(frame_path)
            stem = Path(frame_path).stem
            found = None

            direct = by_name.get(fname.lower())
            if direct:
                found = direct

            if found is None:
                for candidate_stem in (stem.lower(), f"mask_{stem.lower()}"):
                    if candidate_stem in by_stem and by_stem[candidate_stem]:
                        found = sorted(by_stem[candidate_stem])[0]
                        break

            if found is not None:
                self.mask_path_for_image[frame_path] = found

    def _go_to_image(self, idx: int, auto_save: bool = True):
        if not self.image_paths:
            return

        if auto_save and self.current_index >= 0 and self.canvas.mask is not None and self._dirty:
            self._save_mask(silent=True)

        idx = max(0, min(idx, len(self.image_paths) - 1))
        self.current_index = idx

        frame_path = self.image_paths[idx]
        frame_name = os.path.basename(frame_path)

        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if img is None:
            self.statusBar().showMessage(f"Failed to load {frame_name}", 5000)
            return

        self.canvas.cv_image = img
        h, w = img.shape[:2]

        mask = None
        existing_path = self.mask_path_for_image.get(frame_path)
        if existing_path and os.path.exists(existing_path):
            loaded = cv2.imread(existing_path, cv2.IMREAD_GRAYSCALE)
            if loaded is not None:
                if loaded.shape[:2] != (h, w):
                    loaded = cv2.resize(loaded, (w, h), interpolation=cv2.INTER_NEAREST)
                _, loaded = cv2.threshold(loaded, 127, 255, cv2.THRESH_BINARY)
                mask = loaded

        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)

        self.canvas.mask = mask
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False

        # Immediately rebuild texture so current mask_blend is applied
        # (visibility / blend slider persists across navigation)
        self.canvas._img_h, self.canvas._img_w = h, w
        self.canvas.mark_texture_dirty()

        self._img_label.setText(frame_name)
        self._counter_label.setText(f"Image {idx + 1} / {len(self.image_paths)}")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self.image_paths) - 1)
        self._update_mask_info()
        self.statusBar().showMessage(f"Loaded {frame_name}", 2000)
        QTimer.singleShot(50, self.canvas.fit_image)

    def _prev_image(self):
        if self.current_index > 0:
            self._go_to_image(self.current_index - 1)

    def _next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self._go_to_image(self.current_index + 1)

    def push_undo(self):
        if self.canvas.mask is None:
            return
        self._undo_stack.append(self.canvas.mask.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._dirty = True

    def _undo(self):
        if not self._undo_stack or self.canvas.mask is None:
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        self._redo_stack.append(self.canvas.mask.copy())
        self.canvas.mask = self._undo_stack.pop()
        self._dirty = True
        self.canvas.mark_texture_dirty()
        self._update_mask_info()
        self.statusBar().showMessage("Undo", 1000)

    def _redo(self):
        if not self._redo_stack or self.canvas.mask is None:
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        self._undo_stack.append(self.canvas.mask.copy())
        self.canvas.mask = self._redo_stack.pop()
        self._dirty = True
        self.canvas.mark_texture_dirty()
        self._update_mask_info()
        self.statusBar().showMessage("Redo", 1000)

    def _clear_mask(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        self.canvas.mask[:] = 0
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _invert_mask(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        self.canvas.mask = cv2.bitwise_not(self.canvas.mask)
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _dilate_mask(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.canvas.mask = cv2.dilate(self.canvas.mask, kernel, iterations=2)
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _erode_mask(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.canvas.mask = cv2.erode(self.canvas.mask, kernel, iterations=2)
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _smooth_mask(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        blurred = cv2.GaussianBlur(self.canvas.mask, (15, 15), 0)
        _, self.canvas.mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _fill_holes(self):
        if self.canvas.mask is None:
            return
        self.push_undo()
        contours, hierarchy = cv2.findContours(self.canvas.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            filled = self.canvas.mask.copy()
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    cv2.drawContours(filled, contours, i, 255, -1)
            self.canvas.mask = filled
        self.canvas.mark_texture_dirty()
        self._update_mask_info()

    def _save_mask(self, silent: bool = False):
        if self.canvas.mask is None or self.current_index < 0:
            return

        frame_path = self.image_paths[self.current_index]
        frame_name = os.path.basename(frame_path)
        stem = Path(frame_name).stem

        target = self.mask_path_for_image.get(frame_path)
        if not target:
            target = os.path.join(self.masks_folder, f"{stem}.tiff")
            self.mask_path_for_image[frame_path] = target

        os.makedirs(self.masks_folder, exist_ok=True)
        cv2.imwrite(target, self.canvas.mask)
        self._dirty = False
        if not silent:
            self.statusBar().showMessage(f"Saved: {os.path.basename(target)}", 2500)

    def _update_mask_info(self):
        if self.canvas.mask is None:
            self._mask_info_label.setText("No mask")
            return

        total = self.canvas.mask.size
        selected = int(np.count_nonzero(self.canvas.mask))
        pct = (selected / total * 100.0) if total else 0.0
        h, w = self.canvas.mask.shape[:2]
        self._mask_info_label.setText(
            f"Size: {w}×{h}\nSelected: {selected:,} px\nCoverage: {pct:.1f}%"
        )

    def closeEvent(self, event):
        if self._dirty and self.canvas.mask is not None and self.current_index >= 0:
            self._save_mask(silent=True)
        super().closeEvent(event)


def default_data_dir() -> str:
    here = Path(__file__).resolve()
    return str(here.parents[2] / "DATA")


def main():
    parser = argparse.ArgumentParser(description="Interactive tool mask refiner (OpenGL)")
    parser.add_argument("--tool", type=str, default=None, help="Tool ID to open first (e.g. tool002)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir(), help="Path to DATA directory")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA directory not found: {data_dir}")

    # Configure OpenCV multi-threading + OpenCL (Intel Iris Xe)
    _configure_cv_performance()

    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QMainWindow { background-color: #2b2b2b; }
        QWidget { background-color: #2b2b2b; color: #ddd; font-size: 12px; }
        QGroupBox {
            border: 1px solid #555; border-radius: 4px;
            margin-top: 8px; padding-top: 14px;
            font-weight: bold; color: #ccc;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        QPushButton {
            background-color: #3c3c3c; border: 1px solid #555;
            border-radius: 3px; padding: 5px 10px; color: #ddd;
        }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:checked { background-color: #1a6dd4; border-color: #2a8df4; }
        QPushButton:pressed { background-color: #555; }
        QSlider::groove:horizontal {
            height: 6px; background: #555; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #1a6dd4; width: 14px; height: 14px;
            margin: -4px 0; border-radius: 7px;
        }
        QRadioButton { spacing: 5px; }
        QRadioButton::indicator { width: 14px; height: 14px; }
        QStatusBar { background: #1e1e1e; color: #aaa; }
        QCheckBox { spacing: 5px; }
        QLabel { background: transparent; }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 3px 6px;
            color: #ddd;
        }
        """
    )

    window = MaskRefinerWindow(data_dir=data_dir, initial_tool=args.tool)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
