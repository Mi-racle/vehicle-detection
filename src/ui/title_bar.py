from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton


class TitleBarWidget(QWidget):
    def __init__(self, settings: dict, title='Python', parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_alimama_fangyuan_semibold = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_alimama_fangyuan_semibold']))[0])

        self.setFixedSize(1920, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(settings['title_bar_ss'])

        self.icon_label = QLabel(self)
        self.icon_label.setGeometry(20, 14, 36, 36)
        self.icon_label.setPixmap(QPixmap(settings['title_bar_icon']))

        self.text_label = QLabel(title, self)
        self.text_label.setGeometry(64, 13, 300, 36)
        self.text_label.setFont(font_alimama_fangyuan_semibold)
        self.text_label.setStyleSheet(settings['text_label_ss'])

        self.close_button = QPushButton(self)
        self.close_button.setGeometry(1875, 19, 26, 26)
        self.close_button.setIcon(QIcon(settings['close_button_icon']))
        self.close_button.clicked.connect(parent.close if parent else self.close)

        self.old_pos = None

    def setTitleIcon(self, p: str):
        self.icon_label.setPixmap(QPixmap(p))

    def setTitleText(self, t: str):
        self.text_label.setText(t)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.window().move(self.window().pos() + delta)
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None
