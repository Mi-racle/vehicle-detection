import ctypes
import logging
import sys
import threading
from datetime import timedelta
from threading import Thread
from time import time, sleep
from typing import Any

import cv2
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ultralytics import YOLO

from db.camera import get_camera_by_camera_id
from db.group import get_group_by_group_id
from db.model import get_model_by_model_id
from db.result import insert_result
from db.task_offline import update_offline_task_status_by_id
from detectors import ParkingDetector, WrongwayDetector
from ui.display_window import DisplayWindow
from utils import is_in_analysis, get_url_type


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
        cap_in = cv2.VideoCapture(task_entry['file_url'])
        fps = cap_in.get(cv2.CAP_PROP_FPS)

        det_models: dict[str, dict[str, Any]] = {}
        detectors = {}  # {'group_id': Detector}
        camera_entries = {}  # {'group_id': camera_entry}
        model_entries = {}  # {'group_id': model_entry}

        for group_id in task_entry['group_id']:
            group_entry = get_group_by_group_id(group_id)

            camera_entry = get_camera_by_camera_id(group_entry['camera_id'])
            model_entry = get_model_by_model_id(group_entry['model_id'])
            det_args = group_entry['args']

            camera_entries[group_id] = camera_entry
            model_entries[group_id] = model_entry

            if model_entry['file_path'] not in det_models.keys():
                det_models[model_entry['file_path']] = {
                    'model': YOLO(model_entry['file_path']),  # TODO
                    'groups': [group_id],
                    'classes': det_args['cls_indices'] if det_args.get('cls_indices') else None
                }
            else:
                det_models[model_entry['file_path']]['groups'].append(group_id)
                if det_args.get('cls_indices'):
                    classes = det_models[model_entry['file_path']]['classes']
                    det_models[model_entry['file_path']]['classes'] = list(set(classes + det_args['cls_indices']))

            if not model_entry:
                logging.info('Model does not exist')
                update_offline_task_status_by_id(task_entry['id'], -1)
                cap_in.release()
                return
            elif 'parking' in model_entry['model_name'].lower():
                detectors[group_id] = ParkingDetector(fps=fps, **det_args)
            elif 'wrongway' in model_entry['model_name'].lower():
                detectors[group_id] = WrongwayDetector(fps=fps, **det_args)
            else:
                logging.info('Unknown error occurred')
                update_offline_task_status_by_id(task_entry['id'], -1)
                cap_in.release()
                return

        width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_out = cv2.VideoWriter('demo.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

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

                plotted_frame = frame.copy()

                if is_in_analysis(timer, task_entry['analysis_start_time'], task_entry['analysis_end_time']):
                    results = {}  # {group_id: result}

                    for values in det_models.values():
                        det_model = values['model']

                        if isinstance(det_model, YOLO):
                            result = det_model.track(
                                source=frame,
                                imgsz=width,
                                save=False,
                                agnostic_nms=True,
                                persist=True,
                                verbose=False,
                                device=0,
                                classes=values['classes']
                            )[0].cpu().numpy()
                        else:  # TODO OCR
                            result = None

                        for group in values['groups']:
                            results[group] = result

                    print(f'Model cost: {time() - st1:.3f} s')

                    for group in detectors:
                        detector = detectors[group]
                        camera = camera_entries[group]
                        model = model_entries[group]

                        det_ret = detector.update(results[group])
                        plotted_frame = detector.plot(det_ret, plotted_frame)
                        dests = detector.output_corpus(output_dir)

                        if not dests:
                            continue

                        for dest in dests:
                            td_duration_threshold = timedelta(seconds=det_args['duration_threshold'])
                            td_half_diff = timedelta(
                                seconds=max((det_args['video_length'] - det_args['duration_threshold']) // 2, 0))

                            entry = [
                                model['model_name'],
                                model['model_version'],
                                camera['type'],
                                camera['camera_id'],
                                get_url_type(camera['url']),
                                camera['url'],
                                dest,
                                timer - td_half_diff - td_duration_threshold,
                                timer - td_half_diff,
                                None,
                                []  # TODO
                            ]
                            self.display_window.add_info(entry)
                            insert_result(entry)

                cap_out.write(plotted_frame)
                self.display_window.set_image(plotted_frame)

                timer += timedelta(seconds=1 / fps)

                total_cost = time() - st0

                print(f'Total cost: {total_cost:.3f} s')
                print('---------------------')

                sleep(max(1 / fps - total_cost, 0))

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
