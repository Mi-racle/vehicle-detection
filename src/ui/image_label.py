import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QLabel


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.umat = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap = QPixmap().scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.setPixmap(self.pixmap)

    def set_umat(self, umat):
        self.umat = umat
        img_rgb = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg).scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.setPixmap(self.pixmap)

    def resizeEvent(self, event):
        if not self.pixmap.isNull():
            self.umat = cv2.resize(self.umat, [self.size().width(), self.size().height()])
            self.set_umat(self.umat)

        super().resizeEvent(event)
