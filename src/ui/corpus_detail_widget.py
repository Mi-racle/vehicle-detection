from PyQt6.QtWidgets import QWidget, QTableWidget, QHeaderView, QVBoxLayout, QHBoxLayout, QTableWidgetItem

from db import RESULT_DAO


class CorpusDetailWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(['字段', '数值'])
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setAlternatingRowColors(True)
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout = QHBoxLayout()
        layout.addWidget(self.info_table)

        self.setLayout(layout)

        self.infos = []
        self.fields = RESULT_DAO.get_result_header()[1:]
        self.info_table.setRowCount(len(self.fields))

    def add_info(self, info: list):
        self.infos.append(info)
        self.combobox_label.setText(f'选择记录：（目前记录数{len(self.infos)}）')
        self.object_combobox.addItem(info[6])  # info[6] is dest

    def update_table(self):
        info = self.infos[self.object_combobox.currentIndex()]

        for i, pair in enumerate(zip(self.fields, info)):
            self.info_table.setItem(i, 0, QTableWidgetItem(str(pair[0])))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(pair[1])))

