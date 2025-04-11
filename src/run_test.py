import argparse
import logging
import os
import sys
from datetime import datetime

import cv2
import yaml
from PyQt6.QtWidgets import QApplication

from db import TASK_OFFLINE_DAO, TASK_ONLINE_DAO
from ui.main_window import MainWindow


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
    sys_config = yaml.safe_load(open(f'configs/sys_config.yaml', 'r'))

    init_log('logs')

    app = QApplication(sys.argv)
    window = MainWindow(sys_config['output_dir'], sys_config['online'])

    window.show()
    window.func()

    sys.exit(app.exec())
