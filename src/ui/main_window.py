import ctypes
import logging
import os.path
import sys
import threading
from datetime import timedelta
from threading import Thread
from time import time, sleep
from typing import Any

import cv2
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ultralytics import YOLO

from db import CAMERA_DAO, GROUP_DAO, MODEL_DAO, RESULT_DAO, TASK_OFFLINE_DAO
from detectors import ParkingDetector, WrongwayDetector, LanechangeDetector, SpeedingDetector, VelocityDetector, \
    PimDetector, SectionDetector, VolumeDetector, DensityDetector, QueueDetector, JamDetector, PlateDetector, \
    TriangleDetector, SizeDetector
from ocr.ocrer import OCRer
from ui.display_window import DisplayWindow
from utils import in_analysis, get_file_by_substr


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        user32 = ctypes.windll.user32
        screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        self.setWindowTitle('交通态势识别')
        self.setGeometry(0, 0, screen_width, screen_height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.display_window = DisplayWindow()

        layout = QVBoxLayout()
        layout.addWidget(self.display_window)

        self.central_widget.setLayout(layout)

        self.task: Thread | None = None
        self.is_closed = False

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
        dets_args = {}  # {'group_id': det_args}
        camera_entries = {}  # {'group_id': camera_entry}
        model_entries = {}  # {'group_id': model_entry}

        for group_id in task_entry['group_id']:
            group_entry = GROUP_DAO.get_group_by_group_id(group_id)

            camera_entry = CAMERA_DAO.get_camera_by_camera_id(group_entry['camera_id'])
            model_entry = MODEL_DAO.get_model_by_model_id(group_entry['model_id'])
            det_args = group_entry['args']

            camera_entries[group_id] = camera_entry
            model_entries[group_id] = model_entry
            dets_args[group_id] = det_args

            if model_entry['file_path'] not in det_models:
                mdl = None
                if 'yolo' in model_entry['file_path']:
                    mdl = YOLO(model_entry['file_path'])
                elif 'ocr' in model_entry['file_path'] and os.path.isdir(model_entry['file_path']):
                    det_file = get_file_by_substr(model_entry['file_path'], 'det')
                    rec_file = get_file_by_substr(model_entry['file_path'], 'rec')
                    if det_file and rec_file:
                        mdl = OCRer(
                            det_model_path=os.path.join(model_entry['file_path'], det_file),
                            rec_model_path=os.path.join(model_entry['file_path'], rec_file)
                        )
                if mdl:
                    det_models[model_entry['file_path']] = {
                        'model': mdl,
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
                TASK_OFFLINE_DAO.update_offline_task_status_by_id(task_entry['id'], -1)
                cap_in.release()
                return
            elif 'parking' in model_entry['model_name'].lower():
                detectors[group_id] = ParkingDetector(fps=fps, **det_args)
            elif 'wrongway' in model_entry['model_name'].lower():
                detectors[group_id] = WrongwayDetector(fps=fps, **det_args)
            elif 'lanechange' in model_entry['model_name'].lower():
                detectors[group_id] = LanechangeDetector(fps=fps, **det_args)
            elif 'speeding' in model_entry['model_name'].lower():
                detectors[group_id] = SpeedingDetector(fps=fps, **det_args)
            elif 'velocity' in model_entry['model_name'].lower():
                detectors[group_id] = VelocityDetector(fps=fps, **det_args)
            elif 'pim' in model_entry['model_name'].lower():
                detectors[group_id] = PimDetector(fps=fps, **det_args)
            elif 'section' in model_entry['model_name'].lower():
                detectors[group_id] = SectionDetector(fps=fps, **det_args)
            elif 'volume' in model_entry['model_name'].lower():
                detectors[group_id] = VolumeDetector(fps=fps, **det_args)
            elif 'density' in model_entry['model_name'].lower():
                detectors[group_id] = DensityDetector(fps=fps, **det_args)
            elif 'queue' in model_entry['model_name'].lower():
                detectors[group_id] = QueueDetector(fps=fps, **det_args)
            elif 'jam' in model_entry['model_name'].lower():
                detectors[group_id] = JamDetector(fps=fps, **det_args)
            elif 'plate' in model_entry['model_name'].lower():
                detectors[group_id] = PlateDetector(fps=fps, **det_args)
            elif 'triangle' in model_entry['model_name'].lower():
                detectors[group_id] = TriangleDetector(fps=fps, **det_args)
            elif 'size' in model_entry['model_name'].lower():
                detectors[group_id] = SizeDetector(fps=fps, **det_args)
            else:
                logging.info('Unknown model found')
                TASK_OFFLINE_DAO.update_offline_task_status_by_id(task_entry['id'], -1)
                cap_in.release()
                return

        width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_out = cv2.VideoWriter('demo.mp4', cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

        def detect():
            end_as_designed = False
            timer = timedelta(seconds=1 / fps)

            if task_entry['analysis_start_time']:
                cap_in.set(cv2.CAP_PROP_POS_MSEC, task_entry['analysis_start_time'].total_seconds() * 1e3)
                timer = timedelta(seconds=task_entry['analysis_start_time'].total_seconds())

            while cap_in.isOpened() and not self.is_closed:
                st0 = time()

                ret, frame = cap_in.read()
                if not ret:
                    end_as_designed = True
                    break

                st1 = time()

                plotted_frame = frame.copy()

                if not in_analysis(timer, task_entry['analysis_start_time'], task_entry['analysis_end_time']):
                    end_as_designed = True
                    break

                results = {}  # {group_id: result}

                for values in det_models.values():
                    det_model = values['model']

                    if isinstance(det_model, YOLO):
                        result = det_model.track(
                            source=frame,
                            imgsz=(min(640 * height // width, height), min(640, width)),
                            save=False,
                            agnostic_nms=True,
                            persist=True,
                            verbose=False,
                            device=0,
                            classes=values['classes']
                        )[0].cpu().numpy()
                    elif isinstance(det_model, OCRer):
                        result = det_model.ocr(frame)
                    else:
                        result = None
                        logging.error('result is None')

                    for group in values['groups']:
                        results[group] = result

                print(f'Model cost: {time() - st1:.3f} s')

                stats_line = 1
                subscript_line = 1
                for group in detectors:  # 'group' is group id
                    detector = detectors[group]
                    camera = camera_entries[group]
                    model = model_entries[group]
                    dargs = dets_args[group]

                    detector.update(results[group])
                    plotted_frame = detector.plot(plotted_frame, stats_line, subscript_line)
                    stats_line, subscript_line = detector.update_line(stats_line, subscript_line)
                    dests = detector.output_corpus(output_dir, frame)

                    if not dests:
                        continue

                    for dest in dests:
                        if 'video_length' not in dargs:
                            start_time = timer
                            end_time = timer
                        elif 'duration_threshold' not in dargs:
                            td_video_length = timedelta(seconds=dargs['video_length'])
                            start_time = timer - td_video_length
                            end_time = timer
                        else:
                            td_video_length = timedelta(seconds=dargs['video_length'])
                            td_diff = timedelta(seconds=max(dargs['video_length'] - dargs['duration_threshold'], 0))
                            start_time = timer - td_video_length
                            end_time = timer - td_diff

                        entry = [
                            model['model_name'],
                            model['model_version'],
                            camera['type'],
                            camera['camera_id'],
                            2,
                            task_entry['file_url'],
                            dest,
                            start_time,
                            end_time,
                            None,
                            []  # TODO
                        ]
                        self.display_window.add_info(entry)
                        RESULT_DAO.insert_result(entry)

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
                TASK_OFFLINE_DAO.update_offline_task_status_by_id(task_entry['id'], 1)

        self.task = threading.Thread(target=detect)
        self.task.start()

    def closeEvent(self, event):
        self.is_closed = True
        self.task.join()
        sys.exit(0)
