import sys
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QSystemTrayIcon, QWidget, QMenu


class Tray(QSystemTrayIcon):
    close_signal = pyqtSignal()

    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setIcon(QIcon(settings['tray_icon']))
        self.setToolTip(parent.windowTitle() if parent else '未命名标题')
        self.activated.connect(self.__on_tray_activated)

        tray_menu = QMenu()
        tray_menu.addAction('显示窗口', self.__show)
        tray_menu.addAction('退出程序', self.__quit)
        tray_menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid gray;
            }
            QMenu::item {
                padding: 6px 30px;
                text-align: center;
            }
            QMenu::item:selected {
                background-color: #DDDDDD;
            }
        """)

        self.setContextMenu(tray_menu)
        self.show()

        self.__settings = settings
        self.__parent = parent

    def __on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger or \
                reason == QSystemTrayIcon.ActivationReason.MiddleClick:
            self.__show()

    def __show(self):
        if self.__parent:
            self.__parent.showNormal()
            self.__parent.activateWindow()

    def __quit(self):
        self.__parent and self.close_signal.emit() or sys.exit(0)
