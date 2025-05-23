import sys

import yaml
from PyQt6.QtGui import QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication

from ui.ui_utils import AnimatedScreen
from utils import filter_kwargs

if __name__ == '__main__':
    sys_config = yaml.safe_load(open(f'configs/sys_config.yaml', 'r'))
    gui_settings: dict = yaml.safe_load(
        open(f'ui/assets/{'on' if sys_config['online'] else 'off'}line/settings.yaml', 'r'))
    screen_settings: dict = gui_settings['splash_screen']

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    splash_screen = AnimatedScreen(QPixmap(screen_settings.get('screen_image')), '正在启动')
    splash_screen.show()

    from ui.main_window import MainWindow
    kwargs = filter_kwargs(MainWindow.__init__, sys_config)
    window = MainWindow(**kwargs)
    # window = TitleBarWidget(gui_settings['title_bar'])
    # window = TaskDetailWidget(gui_settings['task_detail'])
    # window = TaskListWidget(gui_settings['task_list'])
    # window = CorpusListWidget(gui_settings['corpus_list'])
    # window = CorpusDetailWidget(gui_settings['corpus_detail'])
    # window = Tray(gui_settings['tray'])
    # window = ExitDialog(gui_settings['exit_dialog'])
    window.show()

    splash_screen.finish(window)

    if sys_config.get('auto_size'):
        window.resize(QGuiApplication.primaryScreen().availableGeometry().size())

    app.exec()
