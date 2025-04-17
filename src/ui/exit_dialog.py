from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase, QIcon, QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QRadioButton, QCheckBox, QButtonGroup, QWidget


class ExitDialog(QDialog):
    NOT_REMEMBER = 0
    EXIT = 1
    HIDE = 2

    class CustomTitleBar(QWidget):
        def __init__(self, settings: dict, parent: Optional[QWidget] = None):
            super().__init__(parent)

            font_pingfang_sc_bold = QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_pingfang_sc_bold']))[0])

            self.setFixedSize(490, 56)

            self.__title_tag_label = QLabel('温馨提示', self)
            self.__title_tag_label.setGeometry(24, 16, 64, 24)
            self.__title_tag_label.setFont(font_pingfang_sc_bold)
            self.__title_tag_label.setStyleSheet(settings['title_tag_label_ss'])

            self.__close_button = QPushButton(self)
            self.__close_button.setGeometry(449, 15, 26, 26)
            self.__close_button.setIcon(QIcon(settings['close_button_icon']))
            self.__close_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.__close_button.setStyleSheet("""
                    QPushButton { background: none; border: none; padding: 0px; margin: 0px; }
                    QPushButton:hover, QPushButton:pressed { background: none; }
                    QPushButton:focus { outline: none; }
            """)
            self.__close_button.clicked.connect(parent.close if parent else self.close)

            self.__solid_line = QWidget(self)
            self.__solid_line.setGeometry(0, 55, 490, 1)
            self.__solid_line.setObjectName('solidLine')
            self.__solid_line.setStyleSheet(
                f'QWidget#{self.__solid_line.objectName()} {{ {settings['solid_line_ss']} }}')

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

    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_pingfang_sc_boldface = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_pingfang_sc_boldface']))[0])
        font_pingfang_sc_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_pingfang_sc_regular']))[0])
        font_siyuan_cn_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.setFixedSize(490, 369)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.__background = QWidget(self)
        self.__background.setGeometry(0, 0, 490, 369)
        self.__background.setObjectName('dialogBackground')
        self.__background.setStyleSheet(f'QWidget#{self.__background.objectName()} {{ {settings['background_ss']} }}')

        self.__title_bar = ExitDialog.CustomTitleBar(settings['title_bar'], self)
        self.__title_bar.setGeometry(0, 0, 490, 56)

        middle_group = QWidget(self)
        middle_group.setGeometry(24, 72, 442, 186)

        # middle_group BEGIN
        self.__middle_background = QWidget(middle_group)
        self.__middle_background.setGeometry(0, 0, 442, 186)
        self.__middle_background.setObjectName('middleBackground')
        self.__middle_background.setStyleSheet(
            f'QWidget#{self.__middle_background.objectName()} {{ {settings['middle_background_ss']} }}')

        tip_group = QWidget(middle_group)
        tip_group.setGeometry(108, 24, 226, 84)

        # tip_group BEGIN
        self.__tip_icon_label = QLabel(tip_group)
        self.__tip_icon_label.setGeometry(93, 0, 40, 40)
        self.__tip_icon_label.setPixmap(QPixmap(settings['tip_icon']))

        self.__tip_tag_label = QLabel('请选择您希望进行的操作：', tip_group)
        self.__tip_tag_label.setGeometry(0, 64, 226, 20)
        self.__tip_tag_label.setFont(font_siyuan_cn_regular)
        self.__tip_tag_label.setStyleSheet(settings['tip_tag_label_ss'])
        # tip_group END

        options_group = QWidget(middle_group)
        options_group.setGeometry(91, 132, 261, 22)

        # options_group BEGIN
        self.__exit_radio_button = QRadioButton('退出程序', options_group)
        self.__exit_radio_button.setGeometry(0, 0, 102, 22)
        self.__exit_radio_button.setFont(font_pingfang_sc_boldface)
        self.__exit_radio_button.setStyleSheet(f"""
            QRadioButton::indicator {{width: 14px; height: 14px;}}
            QRadioButton::indicator:unchecked {{image: url({settings['radio_button_unchecked_icon']});}}
            QRadioButton::indicator:checked {{image: url({settings['radio_button_checked_icon']});}}
            {settings['radio_button_ss']}
        """)

        self.__tray_radio_button = QRadioButton('最小化到托盘', options_group)
        self.__tray_radio_button.setGeometry(142, 0, 119, 22)
        self.__tray_radio_button.setChecked(True)
        self.__tray_radio_button.setFont(font_pingfang_sc_boldface)
        self.__tray_radio_button.setStyleSheet(f"""
            QRadioButton::indicator {{width: 14px; height: 14px;}}
            QRadioButton::indicator:unchecked {{image: url({settings['radio_button_unchecked_icon']});}}
            QRadioButton::indicator:checked {{image: url({settings['radio_button_checked_icon']});}}
            {settings['radio_button_ss']}
        """)

        self.__button_group = QButtonGroup()
        self.__button_group.addButton(self.__exit_radio_button, ExitDialog.EXIT)
        self.__button_group.addButton(self.__tray_radio_button, ExitDialog.HIDE)
        # options_group END
        # middle_group END

        self.__checkbox = QCheckBox('不再提示', self)
        self.__checkbox.setGeometry(24, 274, 102, 22)
        self.__checkbox.setFont(font_pingfang_sc_boldface)
        self.__checkbox.setStyleSheet(f"""
            QCheckBox::indicator {{width: 14px; height: 14px;}}
            QCheckBox::indicator:unchecked {{image: url({settings['checkbox_unchecked_icon']});}}
            QCheckBox::indicator:checked {{image: url({settings['checkbox_checked_icon']});}}
            {settings['checkbox_ss']}
        """)

        confirmation_group = QWidget(self)
        confirmation_group.setGeometry(24, 317, 442, 40)

        # confirmation_group BEGIN
        self.__cancel_button = QPushButton('取消', confirmation_group)
        self.__cancel_button.setGeometry(0, 0, 215, 40)
        self.__cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.__cancel_button.setFont(font_pingfang_sc_regular)
        self.__cancel_button.setStyleSheet(settings['cancel_button_ss'])
        self.__cancel_button.clicked.connect(self.reject)

        self.__ok_button = QPushButton('确定', confirmation_group)
        self.__ok_button.setGeometry(227, 0, 215, 40)
        self.__ok_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.__ok_button.setFont(font_pingfang_sc_regular)
        self.__ok_button.setStyleSheet(settings['ok_button_ss'])
        self.__ok_button.clicked.connect(self.accept)
        # confirmation_group END

    def closeEvent(self, event):
        self.__reset()
        super().closeEvent(event)
        
    def reject(self):
        self.__reset()
        super().reject()

    def get_choice(self):
        return self.__button_group.checkedId()

    def remember_choice(self):
        return self.__checkbox.isChecked()

    def __reset(self):
        self.__tray_radio_button.setChecked(True)
        self.__checkbox.setChecked(False)
