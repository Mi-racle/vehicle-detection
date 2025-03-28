import argparse
import logging
import os
import sys
from datetime import datetime

import cv2
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=4)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()

    init_log('logs')

    app = QApplication(sys.argv)
    window = MainWindow()

    if args.online:
        task = TASK_ONLINE_DAO.get_online_task_by_id(args.i)
    else:
        task = TASK_OFFLINE_DAO.get_offline_task_by_id(args.i)

    if not task:
        exit(0)

    window.show()
    window.run_task(task, 'runs')

    while True:
        if not window.is_alive():
            window.destroy()
            sys.exit(0)

        cv2.waitKey(1000)
