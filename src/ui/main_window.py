import sys
import threading
from collections import deque
from time import sleep
from typing import Any, Callable

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from db import TASK_ONLINE_DAO, TASK_OFFLINE_DAO
from ui.corpus_detail import CorpusDetailWidget
from ui.corpus_list import CorpusListWidget
from ui.major_display import MajorDisplayWidget
from ui.task_detail import TaskDetailWidget
from ui.task_list import TaskListWidget
from ui.title_bar import TitleBarWidget


class MainWindow(QWidget):
    __append_task_signal = pyqtSignal(str, str)
    __append_corpus_signal = pyqtSignal(dict)

    def __init__(self, output_dir='', online=True):
        super().__init__()

        # user32 = ctypes.windll.user32
        # screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        window_title = f'视觉AI语料库{'在' if online else '离'}线平台'
        gui_settings: dict = yaml.safe_load(open(f'ui/assets/{'on' if online else 'off'}line/settings.yaml', 'r'))
        settings = gui_settings['main_window']

        self.setWindowIcon(QIcon(settings['window_icon']))
        self.setWindowTitle(window_title)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(0, 0, 1920, 1202)

        self.__background = QWidget(self)
        self.__background.setGeometry(0, 0, 1920, 1202)
        self.__background.setObjectName('mainWindowBackground')
        self.__background.setStyleSheet(
            f'QWidget#{self.__background.objectName()} {{background-image: url({settings['background_image']});}}')

        self.__title_bar = TitleBarWidget(gui_settings['title_bar'], window_title, self)
        self.__title_bar.setGeometry(0, 0, 1920, 64)

        self.__task_detail = TaskDetailWidget(gui_settings['task_detail'], self)
        self.__task_detail.setGeometry(20, 84, 450, 655)

        self.__task_list = TaskListWidget(gui_settings['task_list'], self)
        self.__task_list.setGeometry(20, 749, 450, 433)

        self.__major_display = MajorDisplayWidget(gui_settings['major_display'], self)
        self.__major_display.setGeometry(480, 84, 960, 655)

        self.__corpus_list = CorpusListWidget(gui_settings['corpus_list'], self)
        self.__corpus_list.setGeometry(480, 749, 960, 433)

        self.__corpus_detail = CorpusDetailWidget(gui_settings['corpus_detail'], output_dir, self)
        self.__corpus_detail.setGeometry(1450, 84, 450, 1098)

        self.__corpus_list.send_select_signal.connect(self.__corpus_detail.set_corpus)
        self.__corpus_list.send_unselect_signal.connect(self.__corpus_detail.set_corpus)

        self.__output_dir = output_dir
        self.__online = online
        self.__curr_task_entry: dict | None = None
        self.__task_queue = deque(maxlen=100)
        self.__task_queue_lock = threading.Lock()
        self.__add_tasks_thread: QThread | None = None
        self.__run_tasks_thread: QThread | None = None
        self.__closed = False

    def func(self):
        self.__add_tasks_thread = QThread()
        self.__add_tasks_thread.run = lambda: self.__add_tasks()
        self.__append_task_signal.connect(self.__append_task)
        self.__add_tasks_thread.start()

        self.__run_tasks_thread = QThread()
        self.__run_tasks_thread.run = lambda: self.__run_tasks()
        self.__append_corpus_signal.connect(self.__append_corpus)
        self.__run_tasks_thread.start()

    def closeEvent(self, event):
        self.__closed = True

        if self.__add_tasks_thread:
            self.__add_tasks_thread.wait()
        if self.__run_tasks_thread:
            self.__run_tasks_thread.wait()

        sys.exit(0)

    def __add_tasks(self):
        while not self.__closed:
            if self.__online:
                task_entries = TASK_ONLINE_DAO.get_next_online_task()
            else:
                task_entries = TASK_OFFLINE_DAO.get_next_offline_task()

            for task_entry in task_entries:
                if self.__curr_task_entry and self.__curr_task_entry['id'] == task_entry['id']:
                    continue

                repeated = False

                for t in self.__task_queue:
                    if task_entry['id'] == t['id']:
                        repeated = True
                        break

                if not repeated:
                    with self.__task_queue_lock:
                        self.__task_queue.append(task_entry)

                    # TODO need concat date and time
                    analysis_start_time = str(task_entry['analysis_start_time']) if task_entry[
                        'analysis_start_time'] else '0:00:00'
                    self.__append_task_signal.emit(task_entry['task_name'], analysis_start_time)  # analysis start time

            sleep(1)

    def __run_tasks(self):
        from detect import detect

        while not self.__closed:
            if len(self.__task_queue) == 0:
                sleep(1)
                continue

            with self.__task_queue_lock:
                self.__curr_task_entry = self.__task_queue.popleft()
            self.__task_list.remove_task(0)
            self.__task_detail.set_task(self.__curr_task_entry)

            detect(
                self.__curr_task_entry,
                self.__output_dir,
                self.__is_closed,
                self.__append_corpus_signal,
                self.__set_umat
            )

            self.__curr_task_entry = None
            # TODO reset task_detail and major display

    def __is_closed(self):
        return self.__closed

    def __append_task(self, task_name: str, start_time: str):
        self.__task_list.add_task(task_name, start_time)

    def __append_corpus(self, entry: dict):
        self.__corpus_list.add_corpus(entry)

    def __set_umat(self, umat: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.__major_display.set_umat(umat)
