from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout


class TitleBarWidget(QWidget):
    def __init__(self, settings: dict, title='Python', parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_alimama_fangyuan_semibold = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_alimama_fangyuan_semibold']))[0])

        # self.setFixedSize(1920, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(settings['title_bar_ss'])

        self.__icon_label = QLabel()
        self.__icon_label.setFixedSize(36, 36)
        # self.__icon_label.setGeometry(20, 14, 36, 36)
        self.__icon_label.setPixmap(QPixmap(settings['title_bar_icon']))

        self.__text_label = QLabel(title)
        self.__text_label.setFixedSize(300, 36)
        # self.__text_label.setGeometry(64, 13, 300, 36)
        self.__text_label.setFont(font_alimama_fangyuan_semibold)
        self.__text_label.setStyleSheet(settings['text_label_ss'])

        self.__minimize_button = QPushButton()
        self.__minimize_button.setFixedSize(26, 26)
        # self.__minimize_button.setGeometry(1875, 19, 26, 26)
        self.__minimize_button.setIcon(QIcon(settings['minimize_button_icon']))
        self.__minimize_button.clicked.connect(parent.showMinimized if parent else self.showMinimized)

        self.__maximize_button = QPushButton()
        self.__maximize_button.setFixedSize(26, 26)
        # self.__maximize_button.setGeometry(1875, 19, 26, 26)
        self.__maximize_button.setIcon(QIcon(settings['maximize_button_icon']))
        self.__maximize_button.clicked.connect(parent.showMaximized if parent else self.showMaximized)

        self.__close_button = QPushButton()
        self.__close_button.setFixedSize(26, 26)
        # self.__close_button.setGeometry(1875, 19, 26, 26)
        self.__close_button.setIcon(QIcon(settings['close_button_icon']))
        self.__close_button.clicked.connect(parent.close if parent else self.close)

        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 0, 19, 0)

        layout.addWidget(self.__icon_label)
        layout.addWidget(self.__text_label)
        layout.addStretch()
        layout.addWidget(self.__minimize_button)
        layout.addWidget(self.__maximize_button)
        layout.addWidget(self.__close_button)

        self.__old_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.__old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.__old_pos:
            delta = event.globalPosition().toPoint() - self.__old_pos
            self.window().move(self.window().pos() + delta)
            self.__old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.__old_pos = None
