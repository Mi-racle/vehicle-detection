import inspect
import sys

import yaml
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.title_bar import TitleBarWidget
from ui.task_detail import TaskDetailWidget
from ui.task_list import TaskListWidget
from ui.corpus_list import CorpusListWidget
from ui.corpus_detail import CorpusDetailWidget
from ui.tray import Tray
from ui.exit_dialog import ExitDialog


def filter_kwargs(cls, params: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid_keys = sig.parameters.keys()
    return {k: v for k, v in params.items() if k in valid_keys}


if __name__ == '__main__':
    sys_config = yaml.safe_load(open(f'configs/sys_config.yaml', 'r'))
    gui_settings: dict = yaml.safe_load(open(f'ui/assets/{'on' if sys_config['online'] else 'off'}line/settings.yaml', 'r'))

    kwargs = filter_kwargs(MainWindow, sys_config)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow(**kwargs)
    # window = TitleBarWidget(gui_settings['title_bar'])
    # window = TaskDetailWidget(gui_settings['task_detail'])
    # window = TaskListWidget(gui_settings['task_list'])
    # window = CorpusListWidget(gui_settings['corpus_list'])
    # window = CorpusDetailWidget(gui_settings['corpus_detail'])
    # window = Tray(gui_settings['tray'])
    # window = ExitDialog(gui_settings['exit_dialog'])

    window.show()

    app.exec()
