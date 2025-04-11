from typing import Optional, Any

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget

from ui.ui_utils import ImageLabel


class MajorDisplayWidget(QWidget):
    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setFixedSize(960, 655)

        self.__display_label = ImageLabel(parent=self)
        self.__display_label.setGeometry(0, 0, 960, 655)
        self.__display_label.setText('加载中……')
        self.__display_label.setObjectName('displayLabel')
        self.__display_label.setStyleSheet(
            f'ImageLabel#{self.__display_label.objectName()} {{ {settings['display_label_ss']} }}')

    def set_umat(self, umat: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.__display_label.setUmat(umat)
