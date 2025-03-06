import ctypes
import sys
import threading
from datetime import timedelta
from threading import Thread
from time import time

import cv2
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ultralytics import YOLO

from db.camera import get_camera_by_camera_id
from db.group import get_group_by_group_id
from db.model import get_model_by_model_id
from db.result import insert_result
from db.task_offline import update_offline_task_status_by_id
from detectors import ParkingDetector
from ui.display_window import DisplayWindow
from utils import is_in_analysis, is_url


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        self.setWindowTitle('交通态势识别')
        self.setGeometry(100, 100, screen_width // 2, screen_height // 2)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.display_window = DisplayWindow()

        layout = QVBoxLayout()
        layout.addWidget(self.display_window)

        self.central_widget.setLayout(layout)

        self.task: Thread | None = None
        self.is_close = False

    def is_alive(self):
        if self.task and self.task.is_alive():
            return True

        return False

    def run_offline(
            self,
            task_entry: dict,
            output_dir: str
    ):
        group_entry = get_group_by_group_id(task_entry['group_id'])
        camera_entry = get_camera_by_camera_id(group_entry['camera_id'])
        model_entry = get_model_by_model_id(group_entry['model_id'])

        cap_in = cv2.VideoCapture(task_entry['file_url'])
        fps = cap_in.get(cv2.CAP_PROP_FPS)
        width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_out = cv2.VideoWriter('demo.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

        det_args = group_entry['args']

        if not model_entry:
            print('Model does not exist')
            update_offline_task_status_by_id(task_entry['id'], -1)
            return

        elif 'parking' in model_entry['model_name']:
            detector = ParkingDetector(fps=fps, **det_args)
            det_model = YOLO(model_entry['file_path'])

        else:
            print('Unknown error occurred')
            update_offline_task_status_by_id(task_entry['id'], -1)
            return

        def detect():
            end_as_designed = False
            timer = timedelta(seconds=0)

            while cap_in.isOpened() and not self.is_close:
                st0 = time()

                ret, frame = cap_in.read()
                if not ret:
                    end_as_designed = True
                    break

                st1 = time()

                result = det_model.track(
                    source=frame,
                    imgsz=width,
                    save=False,
                    agnostic_nms=True,
                    persist=True,
                    verbose=False,
                    device=0,
                    classes=det_args['cls_indices']
                )[0].cpu().numpy()

                print(f'Model cost: {time() - st1:.3f} ms')

                plotted_frame = None

                if is_in_analysis(timer, task_entry['analysis_start_time'], task_entry['analysis_end_time']):
                    det_ret = detector.update(result)
                    plotted_frame = detector.plot(det_ret)
                    dests = detector.output_corpus(output_dir)

                    if dests:
                        for dest in dests:
                            td_duration_threshold = timedelta(seconds=det_args['duration_threshold'])
                            td_half_diff = timedelta(
                                seconds=max((det_args['video_length'] - det_args['duration_threshold']) // 2, 0))

                            entry = [
                                model_entry['model_name'],
                                model_entry['model_version'],
                                camera_entry['type'],
                                camera_entry['camera_id'],
                                is_url(camera_entry['url']),
                                camera_entry['url'],
                                dest,
                                timer - td_half_diff - td_duration_threshold,
                                timer - td_half_diff,
                                None,
                                []  # TODO
                            ]
                            self.display_window.add_info(entry)
                            insert_result(entry)

                print(f'Total cost: {time() - st0:.3f} ms')
                print('---------------------')

                timer += timedelta(seconds=1 / fps)

                if plotted_frame is not None:
                    cap_out.write(plotted_frame)
                    self.display_window.set_image(plotted_frame)

                else:
                    cap_out.write(frame)
                    self.display_window.set_image(frame)

            cap_in.release()
            cap_out.release()

            if end_as_designed:
                update_offline_task_status_by_id(task_entry['id'], 1)

        self.task = threading.Thread(target=detect)
        self.task.start()

    def closeEvent(self, event):
        self.is_close = True
        self.task.join()
        sys.exit(0)
