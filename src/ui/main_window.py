import sys
import threading
from collections import deque
from datetime import timedelta, datetime
from time import sleep
from typing import Any

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from PyQt6.QtGui import QIcon, QResizeEvent, QGuiApplication, QCloseEvent, QMoveEvent
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
    __BENCH_HEIGHT = 1102

    def __init__(
            self,
            width=1920,
            height=1102,
            output_dir='',
            online=True,
            use_gpu=False,
            output_to_sql=False,
            output_to_obs=False
    ):
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
        self.__title_bar.setGeometry(0, 0, 1920, 48)

        self.__task_detail = TaskDetailWidget(gui_settings['task_detail'], self)
        self.__task_detail.setGeometry(20, 68, 450, 655)

        self.__task_list = TaskListWidget(gui_settings['task_list'], self)
        self.__task_list.setGeometry(20, 733, 450, 349)

        self.__major_display = MajorDisplayWidget(gui_settings['major_display'], self)
        self.__major_display.setGeometry(480, 68, 960, 655)

        self.__corpus_list = CorpusListWidget(gui_settings['corpus_list'], self)
        self.__corpus_list.setGeometry(480, 733, 960, 349)

        self.__corpus_detail = CorpusDetailWidget(gui_settings['corpus_detail'], output_dir, self)
        self.__corpus_detail.setGeometry(1450, 68, 450, 1014)

        self.__tray = Tray(gui_settings['tray'], self)

        self.__exit_dialog = ExitDialog(gui_settings['exit_dialog'])

        self.__title_bar.maximize_signal.connect(self.__maximize_or_revert)
        self.__task_detail.add_detection_signal.connect(self.__task_detail.add_detection)
        self.__task_detail.reset_detection_signal.connect(self.__task_detail.reset_detection)
        self.__corpus_list.send_select_signal.connect(self.__corpus_detail.set_corpus)
        self.__corpus_list.send_unselect_signal.connect(self.__corpus_detail.set_corpus)
        self.__tray.close_signal.connect(self.__safe_close)

        # self.__old_size = (width, height)
        self.__old_size = QSize(width, height)
        self.__old_pos = QPoint(self.pos())
        self.__output_dir = output_dir
        self.__online = online
        self.__use_gpu = use_gpu
        self.__output_to_sql = output_to_sql
        self.__output_to_obs = output_to_obs
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

    def __maximize_or_revert(self):
        available_rect = QGuiApplication.primaryScreen().availableGeometry()
        max_width = available_rect.width()
        max_height = available_rect.height()

        if self.x() == 0 and self.y() == 0 and self.width() == max_width and self.height() == max_height:  # to revert
            temp_size = self.size()
            temp_pos = self.pos()
            self.resize(self.__old_size)
            self.move(self.__old_pos)
            self.__old_size = temp_size
            self.__old_pos = temp_pos

        else:  # to maximize
            self.__old_size = self.size()
            self.__old_pos = self.pos()
            self.setGeometry(0, 0, max_width, max_height)

    def resizeEvent(self, event: QResizeEvent):
        old_size = event.oldSize()
        new_size = event.size()

        if old_size.width() == -1 or old_size.height() == -1:  # init
            self.__old_size = new_size
            width_diff = self.__BENCH_WIDTH - new_size.width()
            height_diff = self.__BENCH_HEIGHT - new_size.height()

        else:
            self.__old_size = old_size
            width_diff = old_size.width() - new_size.width()
            height_diff = old_size.height() - new_size.height()

        self.__background_label.resize(new_size)
        self.__title_bar.resize(new_size.width(), self.__title_bar.height())
        self.__task_detail.setFixedHeight(self.__task_detail.height() - height_diff)
        self.__task_list.move(self.__task_list.x(), self.__task_list.y() - height_diff)
        self.__major_display.resize(
            self.__major_display.width() - width_diff,
            self.__major_display.height() - height_diff)
        self.__corpus_list.move(self.__corpus_list.x(), self.__corpus_list.y() - height_diff)
        self.__corpus_list.setFixedWidth(self.__corpus_list.width() - width_diff)
        self.__corpus_detail.move(self.__corpus_detail.x() - width_diff, self.__corpus_detail.y())
        self.__corpus_detail.setFixedHeight(self.__corpus_detail.height() - height_diff)

        super().resizeEvent(event)

    def moveEvent(self, event: QMoveEvent):
        self.__old_pos = event.oldPos()

        super().moveEvent(event)

    def closeEvent(self, event: QCloseEvent):
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

            self.__output_to_sql and RESULT_DAO.insert_result(corpus)  # Update sql
            self.__output_to_obs and OBS_DAO.upload_file(f'{self.__output_dir}/{corpus['dest']}')  # Update obs

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
