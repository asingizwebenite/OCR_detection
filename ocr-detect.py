import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageQt
import pytesseract
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QComboBox, QSpinBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QRect, QTimer, QPoint

class ImageLabel(QLabel):
    """ QLabel subclass to show image and support ROI drawing/dragging """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._pix = None
        self.start = None
        self.end = None
        self.roi_rect = None
        self.dragging = False
        self.scale = 1.0  # scale between displayed image and actual image

        self.overlay_boxes = []  # list of (rect, text)

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self._pix = pixmap

    def mousePressEvent(self, event):
        if self._pix is None: return
        if event.button() == Qt.LeftButton:
            self.start = event.pos()
            self.end = event.pos()
            self.dragging = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._pix is None: return
        if self.dragging:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._pix is None: return
        if event.button() == Qt.LeftButton and self.dragging:
            self.end = event.pos()
            self.dragging = False
            self.roi_rect = self.get_image_rect_from_display_rect(self.get_display_rect())
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pix is None:
            return
        qp = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 2)
        qp.setPen(pen)

        # Draw current drag rectangle
        if self.start and self.end:
            rect = self.get_display_rect()
            qp.drawRect(rect)

        # Draw ROI (if selected)
        if self.roi_rect:
            # convert roi_rect (image coords) back to display coords
            display_roi = self.get_display_rect_from_image_rect(self.roi_rect)
            pen2 = QPen(QColor(255, 165, 0), 2)
            qp.setPen(pen2)
            qp.drawRect(display_roi)

        # Draw overlay boxes and text (from OCR)
        if self.overlay_boxes:
            pen3 = QPen(QColor(255, 0, 0), 2)
            qp.setPen(pen3)
            for (box, text) in self.overlay_boxes:
                # box in image coords
                drect = self.get_display_rect_from_image_rect(box)
                qp.drawRect(drect)
                qp.drawText(drect.topLeft() + QPoint(2, -2), text)

    def get_display_rect(self):
        """Return QRect in display coords from start/end points"""
        if not self.start or not self.end:
            return QRect()
        x1 = min(self.start.x(), self.end.x())
        y1 = min(self.start.y(), self.end.y())
        x2 = max(self.start.x(), self.end.x())
        y2 = max(self.start.y(), self.end.y())
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def get_image_rect_from_display_rect(self, display_rect: QRect):
        """Map a rect on the widget (display) back to image pixel coordinates"""
        if self._pix is None or display_rect.width() == 0 or display_rect.height() == 0:
            return None
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self._pix.width(), self._pix.height()
        # We assume the pixmap is scaled to fit label while keeping aspect ratio.
        # Find actual top-left offset (centered)
        scaled = self._pix.size()
        sx, sy = scaled.width(), scaled.height()
        offset_x = (label_w - sx)//2
        offset_y = (label_h - sy)//2
        # convert
        dx1 = display_rect.x() - offset_x
        dy1 = display_rect.y() - offset_y
        dx2 = dx1 + display_rect.width()
        dy2 = dy1 + display_rect.height()
        # clamp
        dx1 = max(0, dx1)
        dy1 = max(0, dy1)
        dx2 = min(sx, dx2)
        dy2 = min(sy, dy2)
        if dx2 <= dx1 or dy2 <= dy1:
            return None
        # map to original image coords (scaled -> image)
        scale_x = pix_w / sx
        scale_y = pix_h / sy
        ix1, iy1 = int(dx1 * scale_x), int(dy1 * scale_y)
        ix2, iy2 = int(dx2 * scale_x), int(dy2 * scale_y)
        return (ix1, iy1, ix2 - ix1, iy2 - iy1)

    def get_display_rect_from_image_rect(self, image_rect):
        """Map image rect (x,y,w,h) to display coords QRect"""
        if self._pix is None or image_rect is None:
            return QRect()
        x, y, w, h = image_rect
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self._pix.width(), self._pix.height()
        sx, sy = self._pix.size().width(), self._pix.size().height()
        offset_x = (label_w - sx)//2
        offset_y = (label_h - sy)//2
        scale_x = sx / pix_w
        scale_y = sy / pix_h
        dx = int(x * scale_x) + offset_x
        dy = int(y * scale_y) + offset_y
        dw = int(w * scale_x)
        dh = int(h * scale_y)
        return QRect(dx, dy, dw, dh)

    def clear_roi(self):
        self.roi_rect = None
        self.start = self.end = None
        self.overlay_boxes = []
        self.update()

class PrintedTextScanner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner (PyTesseract + PyQt5)")
        self.img = None  # current image in BGR (cv2)
        self.display_img = None  # QPixmap
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._query_camera_frame)
        self.is_camera_on = False

        self.init_ui()

    def init_ui(self):
        # Left: image preview
        self.image_label = ImageLabel(self)
        self.image_label.setFixedSize(800, 500)
        self.image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")

        # Right: controls + text output
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        cam_btn = QPushButton("Start Camera")
        cam_btn.clicked.connect(self.toggle_camera)
        self.cam_btn = cam_btn

        capture_btn = QPushButton("Capture Frame")
        capture_btn.clicked.connect(self.capture_frame)

        ocr_btn = QPushButton("Run OCR")
        ocr_btn.clicked.connect(self.run_ocr)

        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self.image_label.clear_roi)

        save_text_btn = QPushButton("Save Extracted Text")
        save_text_btn.clicked.connect(self.save_text)

        self.preproc_combo = QComboBox()
        self.preproc_combo.addItems(["None", "Grayscale", "Binarize(Otsu)", "Adaptive Thresh", "Denoise + Binarize"])

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["eng"])  # add more languages if tesseract language packs installed

        self.oem_spin = QSpinBox()
        self.oem_spin.setRange(0, 3)
        self.oem_spin.setValue(3)
        self.oem_spin.setPrefix("OEM=")

        self.psm_spin = QSpinBox()
        self.psm_spin.setRange(0, 13)
        self.psm_spin.setValue(3)
        self.psm_spin.setPrefix("PSM=")

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(False)
        self.text_output.setPlaceholderText("Extracted text will appear here.")

        # Layouts
        right_layout = QVBoxLayout()
        right_layout.addWidget(load_btn)
        right_layout.addWidget(cam_btn)
        right_layout.addWidget(capture_btn)
        right_layout.addWidget(self.preproc_combo)
        right_layout.addWidget(self.lang_combo)
        right_layout.addWidget(self.oem_spin)
        right_layout.addWidget(self.psm_spin)
        right_layout.addWidget(ocr_btn)
        right_layout.addWidget(clear_roi_btn)
        right_layout.addWidget(save_text_btn)
        right_layout.addWidget(QLabel("OCR output:"))
        right_layout.addWidget(self.text_output)
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(300)
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)
        self.setMinimumSize(1150, 600)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        # read with cv2 (BGR)
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        self.set_image(bgr)

    def set_image(self, bgr):
        self.img = bgr.copy()
        self._update_display_from_bgr(self.img)
        self.image_label.overlay_boxes = []
        self.image_label.clear_roi()
        self.text_output.clear()

    def _update_display_from_bgr(self, bgr):
        # Convert to RGB QImage and then QPixmap scaled to fit label
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label._pix = scaled  # keep scaled pixmap for drawing mapping
        self.display_img = pix

    def toggle_camera(self):
        if not self.is_camera_on:
            # open camera 0
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                QMessageBox.warning(self, "Camera", "Cannot open camera.")
                return
            self.is_camera_on = True
            self.cam_btn.setText("Stop Camera")
            self.timer.start(30)
        else:
            self.timer.stop()
            if self.capture:
                self.capture.release()
            self.is_camera_on = False
            self.cam_btn.setText("Start Camera")

    def _query_camera_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        # show frame as preview (do not override img until capture)
        self._update_display_from_bgr(frame)

    def capture_frame(self):
        if not self.capture or not self.capture.isOpened():
            QMessageBox.information(self, "Capture", "Camera not running. Start camera first.")
            return
        ret, frame = self.capture.read()
        if not ret:
            QMessageBox.warning(self, "Capture", "Failed to capture frame.")
            return
        # set captured frame as current image
        self.set_image(frame)

    def preprocess_for_ocr(self, img_bgr):
        mode = self.preproc_combo.currentText()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if mode == "None":
            return img_bgr
        if mode == "Grayscale":
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mode == "Binarize(Otsu)":
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        if mode == "Adaptive Thresh":
            th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,11,2)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        if mode == "Denoise + Binarize":
            den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        return img_bgr

    def run_ocr(self):
     if self.img is None:
         QMessageBox.warning(self, "OCR", "No image loaded or captured.")
         return

     # Crop ROI if selected
     if self.image_label.roi_rect:
        x, y, w, h = self.image_label.roi_rect
        ocr_img = self.img[y:y+h, x:x+w]
     else:
         ocr_img = self.img.copy()

     # Preprocess
     proc = self.preproc_combo.currentText()
     gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
 
     if proc == "Grayscale":
         pre_img = gray
     elif proc == "Binarize(Otsu)":
         _, pre_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     elif proc == "Adaptive Thresh":
         pre_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
     elif proc == "Denoise + Binarize":
         den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
         _, pre_img = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     else:
         pre_img = ocr_img

    # Convert to PIL
     pil_img = Image.fromarray(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))

    # Tesseract config
     oem = self.oem_spin.value()
     psm = self.psm_spin.value()
     lang = self.lang_combo.currentText()
     config = f"--oem {oem} --psm {psm}"
 
    # Run OCR
     data = pytesseract.image_to_data(pil_img, lang=lang, config=config, output_type=pytesseract.Output.DICT)

    # Extract text & overlay boxes
     extracted_text = []
     overlay_boxes = []

     for i in range(len(data['text'])):
         txt = data['text'][i].strip()
         try:
             conf = int(data['conf'][i])
         except:
             conf = -1

         if txt != "" and conf > 30:
             extracted_text.append(txt)
             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
             # Map to full image if ROI used
             if self.image_label.roi_rect:
                 rx, ry, _, _ = self.image_label.roi_rect
                 box = (rx + x, ry + y, w, h)
             else:
                 box = (x, y, w, h)
             overlay_boxes.append((box, txt))

    # Show text
     self.text_output.setPlainText(" ".join(extracted_text))
     self.image_label.overlay_boxes = overlay_boxes
     self.image_label.update()

    def save_text(self):
        txt = self.text_output.toPlainText()
        if not txt.strip():
            QMessageBox.information(self, "Save", "No text to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save extracted text", "extracted_text.txt", "Text files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        QMessageBox.information(self, "Saved", f"Saved to {path}")

def main():
    app = QApplication(sys.argv)
    win = PrintedTextScanner()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
