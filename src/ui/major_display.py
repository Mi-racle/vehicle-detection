from typing import Optional, Any

import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QFont, QFontDatabase, QResizeEvent
from PyQt6.QtWidgets import QWidget, QLabel, QGridLayout

from ui.ui_utils import ImageLabel


class MajorDisplayWidget(QWidget):
    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_siyuan_cn_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.__background = QLabel(self)
        self.__background.setGeometry(0, 0, 960, 655)
        self.__background.setPixmap(QPixmap(settings['background_image']))
        self.__background.setScaledContents(True)

        self.__display_label = ImageLabel(parent=self)
        self.__display_label.setGeometry(0, 0, 960, 655)

        loading_group = QWidget()
        loading_group.setFixedSize(180, 40)

        # loading_group BEGIN
        self.__loading_icon_label = QLabel(loading_group)
        self.__loading_icon_label.setGeometry(0, 0, 40, 40)
        self.__loading_icon_label.setPixmap(QPixmap(settings['loading_icon']))

        self.__loading_tag_label = QLabel('加载中', loading_group)
        self.__loading_tag_label.setGeometry(48, 11, 132, 18)
        self.__loading_tag_label.setFont(font_siyuan_cn_regular)
        self.__loading_tag_label.setStyleSheet(settings['loading_tag_label_ss'])
        # loading_group END

        layout = QGridLayout(self)
        layout.setColumnStretch(0, 455)
        layout.setColumnStretch(2, 400)
        layout.setRowStretch(0, 308)
        layout.setRowStretch(2, 307)

        layout.addWidget(loading_group, 1, 1)

    def resizeEvent(self, event: QResizeEvent):
        if event.oldSize().width() != -1 and event.oldSize().height() != -1:
            self.__background.resize(event.size())
            self.__display_label.resize(event.size())

        super().resizeEvent(event)

    def set_umat(self, umat: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.__loading_icon_label.setVisible(False)
        self.__loading_tag_label.setVisible(False)
        self.__display_label.setUmat(umat)

    def reset(self):
        self.__loading_icon_label.setVisible(True)
        self.__loading_tag_label.setVisible(True)
        self.__loading_tag_label.setText('等待任务中')
        self.__display_label.reset()
