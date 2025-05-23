import logging
import os
import sys
from datetime import datetime

import yaml
from PyQt6.QtGui import QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication

from ui.ui_utils import AnimatedScreen
from utils import filter_kwargs


def init_log(dir_name: str):
    os.makedirs(dir_name, exist_ok=True)

    log_filename = f'{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    log_file_path = os.path.join(dir_name, log_filename)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


if __name__ == '__main__':
    sys_config: dict = yaml.safe_load(open('configs/sys_config.yaml', 'r'))
    screen_settings: dict = yaml.safe_load(
        open(f'ui/assets/{'on' if sys_config['online'] else 'off'}line/settings.yaml', 'r'))['splash_screen']

    init_log(sys_config['log_dir'])

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    splash_screen = AnimatedScreen(QPixmap(screen_settings.get('screen_image')), '正在启动')
    splash_screen.show()

    from ui.main_window import MainWindow
    kwargs = filter_kwargs(MainWindow.__init__, sys_config)
    window = MainWindow(**kwargs)
    window.show()

    splash_screen.finish(window)

    if sys_config.get('auto_size'):  # no 'auto_size' or auto_size == False
        window.resize(QGuiApplication.primaryScreen().availableGeometry().size())

    window.func()

    sys.exit(app.exec())
