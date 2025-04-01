from typing import Any

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QTableWidget, QHeaderView, QTableWidgetItem, QVBoxLayout, QComboBox, \
    QLabel

from db import RESULT_DAO
from ui.image_label import ImageLabel


class DisplayWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        self.image_label = ImageLabel()
        # self.image_label.setText('显示区')
        self.image_label.setStyleSheet("QLabel {background-color: grey;}")

        right_widget = QWidget()
        right_widget.setFixedWidth(200)

        right_layout = QVBoxLayout()

        self.combobox_label = QLabel()
        self.combobox_label.setText('选择记录：（目前记录数0）')

        self.object_combobox = QComboBox()
        self.object_combobox.setFixedWidth(192)
        self.object_combobox.currentIndexChanged.connect(self.update_table)

        self.info_table = QTableWidget()
        self.info_table.setFixedWidth(200)
        self.info_table.setColumnCount(2)
        self.info_table.setRowCount(11)
        self.info_table.setHorizontalHeaderLabels(['字段', '数值'])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        right_layout.addWidget(self.combobox_label)
        right_layout.addWidget(self.object_combobox)
        right_layout.addWidget(self.info_table)

        right_widget.setLayout(right_layout)

        layout.addWidget(self.image_label)
        layout.addWidget(right_widget)
        # layout.addWidget(self.info_table)

        self.setLayout(layout)

        self.infos = []
        self.fields = RESULT_DAO.get_result_header()[1:]

    def set_image(self, frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.image_label.set_umat(frame)

    def add_info(self, info: list):
        self.infos.append(info)
        self.combobox_label.setText(f'选择记录：（目前记录数{len(self.infos)}）')
        self.object_combobox.addItem(info[6])  # info[6] is dest

    def update_table(self):
        info = self.infos[self.object_combobox.currentIndex()]

        for i, pair in enumerate(zip(self.fields, info)):
            self.info_table.setItem(i, 0, QTableWidgetItem(str(pair[0])))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(pair[1])))
