from typing import Optional, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget, QLabel

from ui.ui_utils import ImageLabel


class MajorDisplayWidget(QWidget):
    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_siyuan_cn_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.setFixedSize(960, 655)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName('majorDisplayWidget')
        self.setStyleSheet(
            f'{type(self).__name__}#{self.objectName()} {{ background-image: url({settings['background_image']});}}')

        self.__display_label = ImageLabel(parent=self)
        self.__display_label.setGeometry(0, 0, 960, 655)

        loading_group = QWidget(self)
        loading_group.setGeometry(428, 308, 180, 40)

        # loading_group BEGIN
        self.__loading_icon_label = QLabel(loading_group)
        self.__loading_icon_label.setGeometry(0, 0, 40, 40)
        self.__loading_icon_label.setPixmap(QPixmap(settings['loading_icon']))

        self.__loading_tag_label = QLabel('加载中', loading_group)
        self.__loading_tag_label.setGeometry(48, 11, 132, 18)
        self.__loading_tag_label.setFont(font_siyuan_cn_regular)
        self.__loading_tag_label.setStyleSheet(settings['loading_tag_label_ss'])
        # loading_group END

    def set_umat(self, umat: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.__loading_icon_label.setVisible(False)
        self.__loading_tag_label.setVisible(False)
        self.__display_label.setUmat(umat)

    def reset(self):
        self.__loading_icon_label.setVisible(True)
        self.__loading_tag_label.setVisible(True)
        self.__loading_tag_label.setText('等待任务中')
        self.__display_label.reset()
