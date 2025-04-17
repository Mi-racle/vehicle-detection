import inspect
import logging
import os
import sys
from datetime import datetime

import yaml
from PyQt6.QtWidgets import QApplication

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


def filter_kwargs(cls, params: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid_keys = sig.parameters.keys()
    return {k: v for k, v in params.items() if k in valid_keys}


if __name__ == '__main__':
    sys_config: dict = yaml.safe_load(open('configs/sys_config.yaml', 'r'))

    init_log(sys_config['log_dir'])

    kwargs = filter_kwargs(MainWindow, sys_config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow(**kwargs)
    window.show()
    window.func()

    sys.exit(app.exec())
