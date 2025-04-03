import ctypes
import imghdr
import sys
import threading
from collections import deque
from threading import Thread
from time import sleep

import cv2
from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QTableWidget, QHeaderView, QTableWidgetItem, \
    QAbstractItemView

from db import TASK_ONLINE_DAO, TASK_OFFLINE_DAO
from ui.image_label import ImageLabel


class MainWindow(QMainWindow):
    def __init__(self, output_dir='', online=True):
        super().__init__()

        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        self.setWindowTitle('交通态势识别')
        self.setGeometry(0, 0, screen_width, screen_height)
        self.showMaximized()

        self.central_widget = QWidget()

        self.task_detail_table = QTableWidget()
        self.task_detail_table.setColumnCount(2)
        self.task_detail_table.horizontalHeader().setVisible(False)
        self.task_detail_table.verticalHeader().setVisible(False)
        self.task_detail_table.setAlternatingRowColors(True)
        self.task_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.task_detail_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.task_detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.task_fields = ['任务', '摄像头位置', '视频源', '识别内容', '创建时间', '开始时间', '结束时间']
        self.task_detail_table.setRowCount(len(self.task_fields))  # TODO

        self.task_list_table = QWidget()
        self.task_list_table = QTableWidget()
        self.task_list_table.setColumnCount(2)
        self.task_list_table.setHorizontalHeaderLabels(['任务', '开始时间'])
        self.task_list_table.setAlternatingRowColors(True)
        self.task_list_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.task_list_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.task_list_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.major_display_label = ImageLabel()
        self.major_display_label.setText('加载中……')
        self.major_display_label.setStyleSheet("QLabel {background-color: grey;}")

        self.corpus_list_table = QTableWidget()
        self.corpus_list_table.setColumnCount(5)
        self.corpus_list_table.setHorizontalHeaderLabels(['位置', '模型', '文件', '开始时间', '结束时间'])
        self.corpus_list_table.setAlternatingRowColors(True)
        self.corpus_list_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.corpus_list_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.corpus_list_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.corpus_list_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.corpus_list_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.corpus_list_table.cellClicked.connect(self.__on_corpus_selected)

        self.corpus_detail_table = QTableWidget()
        self.corpus_detail_table.setColumnCount(2)
        self.corpus_detail_table.horizontalHeader().setVisible(False)
        self.corpus_detail_table.verticalHeader().setVisible(False)
        self.corpus_detail_table.setAlternatingRowColors(True)
        self.corpus_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.corpus_detail_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.corpus_detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.corpus_fields = ['文件', '模型', '模型版本', '摄像头类型', '摄像头编号',
                              '视频类型', '视频源', '开始时间', '结束时间', '车牌号', '位置']
        self.corpus_detail_table.setRowCount(len(self.corpus_fields))
        for i, corpus_field in enumerate(self.corpus_fields):
            self.corpus_detail_table.setItem(i, 0, QTableWidgetItem(str(corpus_field)))
            self.corpus_detail_table.setItem(i, 1, QTableWidgetItem('-'))

        self.minor_display_label = ImageLabel()
        self.minor_display_label.setText('尚未选定语料文件')
        self.minor_display_label.setStyleSheet("QLabel {background-color: grey;}")

        layout = QGridLayout()
        layout.addWidget(self.task_detail_table, 0, 0, 6, 4)  # placeholder
        layout.addWidget(self.task_list_table, 6, 0, 3, 4)  # placeholder
        layout.addWidget(self.major_display_label, 0, 4, 6, 8)
        layout.addWidget(self.corpus_list_table, 6, 4, 3, 8)  # placeholder
        layout.addWidget(self.corpus_detail_table, 0, 12, 6, 4)
        layout.addWidget(self.minor_display_label, 6, 12, 3, 4)  # placeholder

        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

        self.output_dir = output_dir
        self.online = online
        self.corpus_entries: list[list] = []
        self.curr_task_entry: dict | None = None
        self.task_queue = deque(maxlen=100)
        self.task_queue_lock = threading.Lock()
        self.task_thread: Thread | None = None
        self.add_tasks_thread: Thread | None = None
        self.run_tasks_thread: Thread | None = None
        self.is_closed = False

    def func(self):
        self.add_tasks_thread = threading.Thread(target=self.__add_tasks)
        self.add_tasks_thread.start()
        self.run_tasks_thread = threading.Thread(target=self.__run_tasks)
        self.run_tasks_thread.start()

    def __add_tasks(self):
        while not self.is_closed:
            if self.online:
                task_entries = TASK_ONLINE_DAO.get_next_online_task()
            else:
                task_entries = TASK_OFFLINE_DAO.get_next_offline_task()

            for task_entry in task_entries:
                if self.curr_task_entry and self.curr_task_entry['id'] == task_entry['id']:
                    continue

                repeated = False

                for t in self.task_queue:
                    if task_entry['id'] == t['id']:
                        repeated = True
                        break

                if not repeated:
                    self.__append_task(task_entry)

            sleep(1)

    def __run_tasks(self):
        from detect import detect

        while not self.is_closed:
            if (self.task_thread and self.task_thread.is_alive()) or len(self.task_queue) == 0:
                sleep(1)
                continue

            with self.task_queue_lock:
                self.curr_task_entry = self.task_queue.popleft()
            self.task_list_table.removeRow(0)
            self.__update_task_detail(self.curr_task_entry)

            self.task_thread = threading.Thread(target=detect, args=(self, self.curr_task_entry, self.output_dir))
            self.task_thread.start()

    def closeEvent(self, event):
        self.is_closed = True

        if self.task_thread:
            self.task_thread.join()
        if self.add_tasks_thread:
            self.add_tasks_thread.join()
        if self.run_tasks_thread:
            self.run_tasks_thread.join()

        sys.exit(0)

    def __append_task(self, task_entry: dict):
        with self.task_queue_lock:
            self.task_queue.append(task_entry)

        row = self.task_list_table.rowCount()
        self.task_list_table.insertRow(row)

        self.task_list_table.setItem(row, 0, QTableWidgetItem(str(task_entry['task_name'])))  # task name
        analysis_start_time = str(task_entry['analysis_start_time']) if task_entry['analysis_start_time'] else '0:00:00'
        self.task_list_table.setItem(row, 1, QTableWidgetItem(analysis_start_time))  # analysis start time

    def append_corpus(self, entry: list):
        self.corpus_entries.append(entry)

        row = self.corpus_list_table.rowCount()
        self.corpus_list_table.insertRow(row)

        self.corpus_list_table.setItem(row, 0, QTableWidgetItem(str(entry[-1])))  # location
        self.corpus_list_table.setItem(row, 1, QTableWidgetItem(f'{entry[0]}v{entry[1]}'))  # model name and version
        self.corpus_list_table.setItem(row, 2, QTableWidgetItem(str(entry[6])))
        self.corpus_list_table.setItem(row, 3, QTableWidgetItem(str(entry[7])))
        self.corpus_list_table.setItem(row, 4, QTableWidgetItem(str(entry[8])))

    def __on_corpus_selected(self, row: int):
        corpus_entry = self.corpus_entries[row].copy()
        corpus_entry.insert(0, corpus_entry.pop(6))
        corpus_entry[3] = '固定位' if corpus_entry[3] == 1 else ('车辆' if corpus_entry[3] == 2 else '无人机')
        corpus_entry[5] = '实时视频流' if corpus_entry[5] == 1 else '视频文件'

        for i, pair in enumerate(zip(self.corpus_fields, corpus_entry)):
            self.corpus_detail_table.setItem(i, 0, QTableWidgetItem(str(pair[0])))
            self.corpus_detail_table.setItem(i, 1, QTableWidgetItem(str(pair[1])))

        corpus_path = f'{self.output_dir}/{corpus_entry[0]}'

        try:
            umat = None

            if imghdr.what(corpus_path):
                umat = cv2.imread(corpus_path)

            else:
                cap = cv2.VideoCapture(corpus_path)
                ret, frame = cap.read()

                if ret:
                    umat = frame

                cap.release()

            if umat is not None:
                self.minor_display_label.set_umat(umat)

        except Exception as e:
            print(e)

    def __update_task_detail(self, task_entry: dict):
        # TODO
        pass
