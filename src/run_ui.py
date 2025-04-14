import sys
from threading import Thread

import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.sip import array

from ui.corpus_detail import CorpusDetailWidget
from ui.corpus_list import CorpusListWidget
from ui.main_window import MainWindow
from ui.task_detail import TaskDetailWidget
from ui.task_list import TaskListWidget
from ui.title_bar import TitleBarWidget

if __name__ == '__main__':
    sys_config = yaml.safe_load(open(f'configs/sys_config.yaml', 'r'))
    gui_settings: dict = yaml.safe_load(open(f'ui/assets/{'on' if sys_config['online'] else 'off'}line/settings.yaml', 'r'))

    app = QApplication(sys.argv)

    # window = MainWindow(sys_config['output_dir'], sys_config['online'])
    # window = TitleBarWidget(gui_settings['title_bar'])
    # window = TaskDetailWidget(gui_settings['task_detail'])
    window = TaskListWidget(gui_settings['task_list'])
    # window = CorpusListWidget(gui_settings['corpus_list'])
    # window = CorpusDetailWidget(gui_settings['corpus_detail'])x

    window.show()
    # Thread(target=window.scroll_test, args=(5,)).start()
    # window.scroll_test(5)
    app.exec()
