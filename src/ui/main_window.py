import sys
import threading
from collections import deque
from datetime import timedelta, datetime
from time import sleep
from typing import Any

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QDialog

from db import TASK_ONLINE_DAO, TASK_OFFLINE_DAO, RESULT_DAO, OBS_DAO
from ui.corpus_detail import CorpusDetailWidget
from ui.corpus_list import CorpusListWidget
from ui.exit_dialog import ExitDialog
from ui.major_display import MajorDisplayWidget
from ui.task_detail import TaskDetailWidget
from ui.task_list import TaskListWidget
from ui.title_bar import TitleBarWidget
from ui.tray import Tray
from ui.ui_utils import ImageLabel


class MainWindow(QWidget):
    __append_task_signal = pyqtSignal(str, str)
    __append_corpus_signal = pyqtSignal(dict, str)
    __reset_display_signal = pyqtSignal()
    __BENCH_WIDTH = 1920
    __BENCH_HEIGHT = 1202

    def __init__(self, width=1920, height=1202, output_dir='', online=True, use_gpu=False):
        super().__init__()

        window_title = f'视觉AI语料库{'在' if online else '离'}线平台'
        gui_settings: dict = yaml.safe_load(open(f'ui/assets/{'on' if online else 'off'}line/settings.yaml', 'r'))
        settings = gui_settings['main_window']

        self.setGeometry(0, 0, width, height)
        self.setWindowIcon(QIcon(settings['window_icon']))
        self.setWindowTitle(window_title)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.__background_label = ImageLabel(settings['background_image'], self)
        self.__background_label.setGeometry(0, 0, width, height)

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

        self.__tray = Tray(gui_settings['tray'], self)

        self.__exit_dialog = ExitDialog(gui_settings['exit_dialog'])

        self.__corpus_list.send_select_signal.connect(self.__corpus_detail.set_corpus)
        self.__corpus_list.send_unselect_signal.connect(self.__corpus_detail.set_corpus)
        self.__tray.close_signal.connect(self.__safe_close)

        self.__output_dir = output_dir
        self.__online = online
        self.__use_gpu = use_gpu
        self.__curr_task_entry: dict | None = None
        self.__curr_task_lock = threading.Lock()
        self.__task_queue = deque(maxlen=100)
        self.__task_queue_lock = threading.Lock()
        self.__corpus_queue = deque(maxlen=1000)
        self.__corpus_queue_lock = threading.Lock()
        self.__add_tasks_thread: QThread | None = None
        self.__run_tasks_thread: QThread | None = None
        self.__upload_corpora_thread: QThread | None = None
        self.__closed = False
        self.__exit_dialog_state = ExitDialog.NOT_REMEMBER

    def func(self):
        self.__add_tasks_thread = QThread()
        self.__add_tasks_thread.run = lambda: self.__add_tasks()
        self.__append_task_signal.connect(self.__append_task)
        self.__add_tasks_thread.start()

        self.__run_tasks_thread = QThread()
        self.__run_tasks_thread.run = lambda: self.__run_tasks()
        self.__append_corpus_signal.connect(self.__append_corpus)
        self.__reset_display_signal.connect(self.__reset_display)
        self.__run_tasks_thread.start()

        self.__upload_corpora_thread = QThread()
        self.__upload_corpora_thread.run = lambda: self.__upload_corpora()
        self.__upload_corpora_thread.start()

    def resizeEvent(self, event):
        qsize = self.size()
        self.__background_label.resize(qsize)
        # TODO self-adapt
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.__exit_dialog_state == ExitDialog.NOT_REMEMBER:
            if self.__exit_dialog.exec() == QDialog.DialogCode.Accepted:
                self.__exit_dialog_state = self.__exit_dialog.get_choice()

                if self.__exit_dialog_state == ExitDialog.EXIT:
                    self.__safe_close()
                elif self.__exit_dialog_state == ExitDialog.HIDE:
                    event.ignore()
                    self.hide()

                if not self.__exit_dialog.remember_choice():
                    self.__exit_dialog_state = ExitDialog.NOT_REMEMBER

            else:
                event.ignore()

        elif self.__exit_dialog_state == ExitDialog.EXIT:
            self.__safe_close()

        elif self.__exit_dialog_state == ExitDialog.HIDE:
            event.ignore()
            self.hide()

    def __safe_close(self):
        self.__closed = True

        self.__add_tasks_thread and self.__add_tasks_thread.wait()
        self.__run_tasks_thread and self.__run_tasks_thread.wait()
        self.__upload_corpora_thread and self.__upload_corpora_thread.wait()

        sys.exit(0)

    def __add_tasks(self):
        while not self.__closed:
            if self.__online:
                task_entries = TASK_ONLINE_DAO.get_next_online_tasks()
            else:
                task_entries = TASK_OFFLINE_DAO.get_next_offline_tasks()

            for task_entry in task_entries:
                with self.__curr_task_lock:
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

                    # start_time only for view
                    start_time = task_entry['analysis_start_time'] or timedelta()
                    if self.__online & ('execute_date' in task_entry):
                        start_time = datetime.combine(task_entry['execute_date'], datetime.min.time()) + start_time
                    self.__append_task_signal.emit(task_entry['task_name'], str(start_time))

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
                self.__online,
                self.__use_gpu,
                self.__is_closed,
                self.__append_corpus_signal,
                self.__set_display
            )

            with self.__curr_task_lock:
                self.__curr_task_entry = None
            self.__task_detail.set_task(None)
            self.__reset_display_signal.emit()

    def __upload_corpora(self):
        while not self.__closed:
            if len(self.__corpus_queue) == 0:
                sleep(1)
                continue

            with self.__corpus_queue_lock:
                corpus = self.__corpus_queue.popleft()

            RESULT_DAO.insert_result(corpus)  # Update sql
            # TODO for test. Need uncomment.
            # OBS_DAO.upload_file(f'{self.__output_dir}/{corpus['dest']}')  # Update obs

    def __is_closed(self):
        return self.__closed

    def __append_task(self, task_name: str, start_time: str):
        self.__task_list.add_task(task_name, start_time)

    def __append_corpus(self, entry: dict, camera_position: str):
        self.__corpus_list.add_corpus(entry, camera_position)
        with self.__corpus_queue_lock:
            self.__corpus_queue.append(entry)

    def __set_display(self, umat: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray):
        self.__major_display.set_umat(umat)

    def __reset_display(self):
        self.__major_display.reset()
