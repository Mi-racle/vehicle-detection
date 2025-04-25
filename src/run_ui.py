import sys

import yaml
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from utils import filter_kwargs

if __name__ == '__main__':
    sys_config = yaml.safe_load(open(f'configs/sys_config.yaml', 'r'))
    gui_settings: dict = yaml.safe_load(open(f'ui/assets/{'on' if sys_config['online'] else 'off'}line/settings.yaml', 'r'))

    kwargs = filter_kwargs(MainWindow, sys_config)

    app = QApplication(sys.argv)
    # app.setQuitOnLastWindowClosed(False)

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
