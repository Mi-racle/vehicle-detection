import logging
from time import sleep
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer, QThread, QDeadlineTimer
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QFont
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QScrollArea, QSplashScreen


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
    def __init__(self, maxlen=100, reverse=False):
        super().__init__()

        self.setStyleSheet('background: transparent;')

        self.__layout = QVBoxLayout(self)
        self.__layout.setSpacing(0)
        self.__layout.setContentsMargins(0, 0, 0, 0)

        self.__maxlen = maxlen
        self.__reverse = reverse
        self.__items: list[QWidget] = []

    def addItem(self, item: QWidget):
        if len(self.__items) >= self.__maxlen:
            item_pop = self.__items.pop(0)
            self.__layout.removeWidget(item_pop)
            item_pop.deleteLater()

        if self.__reverse:
            self.__layout.insertWidget(0, item)

        else:
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
                logging.error(ie)

    def removeAll(self):
        while self.__items:
            self.removeItem(0)

    def getItem(self, object_name: str):
        for item in self.__items:
            if item.objectName() == object_name:
                return item

        return None

    def getItemNum(self):
        return len(self.__items)

    def __updateSize(self):
        width = self.__items and self.__items[0].width() or 0
        height = sum([item.height() for item in self.__items])
        self.setFixedSize(width, height)


class ScrollAreaWithShift(QScrollArea):
    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            delta = event.angleDelta().y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)


class AnimatedScreen(QSplashScreen):
    def __init__(self, pixmap: QPixmap, message_base='', interval_s=0.222, color=Qt.GlobalColor.white):
        super().__init__(pixmap)

        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        self.__message_base = message_base
        self.__interval_s = interval_s
        self.__color = color
        self.__dot_count = 0
        self.__finished = False
        self.__animation_thread: QThread | None = None

    def show(self):
        super().show()

        self.__animation_thread = QThread()
        self.__animation_thread.run = lambda: self.__update_message()
        self.__animation_thread.start()

    def finish(self, w):
        self.__finished = True

        self.__animation_thread and self.__animation_thread.wait(QDeadlineTimer(3 * 1000))

        super().finish(w)

    def __update_message(self):
        while not self.__finished:
            dots = '.' * ((self.__dot_count % 3) + 1)
            self.showMessage(f'{self.__message_base}{dots}', Qt.AlignmentFlag.AlignBottom, self.__color)
            self.__dot_count += 1

            sleep(self.__interval_s)
