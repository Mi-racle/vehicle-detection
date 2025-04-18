from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout


class ImageLabel(QLabel):
    def __init__(self, img: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setPixmap(QPixmap(img).scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio))

        self.__umat = cv2.imread(img) if img else None

    def setUmat(self, umat):
        self.__umat = umat

        if umat is not None:
            img_rgb = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            self.setPixmap(
                QPixmap.fromImage(
                    QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                )
                .scaled(
                    self.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio
                )
            )

        else:
            self.setPixmap(QPixmap())

    def reset(self):
        self.setUmat(None)

    def resizeEvent(self, event):
        if self.__umat is not None:
            self.__umat = cv2.resize(self.__umat, [self.size().width(), self.size().height()])
            self.setUmat(self.__umat)

        super().resizeEvent(event)


class ScrollContainer(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet('background: transparent;')

        self.__layout = QVBoxLayout(self)
        self.__layout.setSpacing(0)
        self.__layout.setContentsMargins(0, 0, 0, 0)

        self.__items: list[QWidget] = []

    def addItem(self, item: QWidget):
        self.__layout.addWidget(item)
        self.__items.append(item)
        self.__updateSize()

    def removeItem(self, index: int):
        if self.__items:
            try:
                item = self.__items.pop(index)
                self.__layout.removeWidget(item)
                item.deleteLater()
                self.__updateSize()

            except IndexError as ie:
                print(ie)

    def getItem(self, object_name: str):
        for item in self.__items:
            if item.objectName() == object_name:
                return item

    def __updateSize(self):
        self.setFixedHeight(sum([item.height() for item in self.__items]))
