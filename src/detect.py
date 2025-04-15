import logging
import os
from datetime import timedelta, datetime
from time import time, sleep
from typing import Any, Callable

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal
from ultralytics import YOLO

from db import CAMERA_DAO, GROUP_DAO, TASK_ONLINE_DAO, TASK_OFFLINE_DAO, MODEL_DAO, RESULT_DAO, OBS_DAO
from detectors import ParkingDetector, WrongwayDetector, LanechangeDetector, SpeedingDetector, VelocityDetector, \
    PimDetector, SectionDetector, VolumeDetector, DensityDetector, QueueDetector, JamDetector, PlateDetector, \
    TriangleDetector, SizeDetector, ObjectDetector
from ocr.ocrer import OCRer
from utils import get_file_by_substr, in_analysis


def detect(
        task_entry: dict,
        output_dir: str,
        is_closed: Callable[[], bool] | None = None,
        append_corpus_signal: pyqtSignal | None = None,
        set_umat: Callable[[cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray], None] | None = None,
):
    is_online = 'file_url' not in task_entry

    if is_online:
        video_url = CAMERA_DAO.get_camera_by_camera_id(
            GROUP_DAO.get_group_by_group_id(
                task_entry['group_id'][0]
            )['camera_id']
        )['url']
        update_task_status = TASK_ONLINE_DAO.update_online_task_status_by_id

    else:
        video_url = task_entry['file_url']
        update_task_status = TASK_OFFLINE_DAO.update_offline_task_status_by_id

    cap_in = cv2.VideoCapture(video_url)
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
            update_task_status(task_entry['id'], -1)
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
        elif 'object' in model_entry['model_name'].lower():
            detectors[group_id] = ObjectDetector(fps=fps, **det_args)
        else:
            logging.info('Unknown model found')
            update_task_status(task_entry['id'], -1)
            cap_in.release()
            return

    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_as_designed = False
    timer = timedelta(seconds=1 / fps)

    if is_online:
        now = datetime.now()
        timer = timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)

    else:
        if task_entry['analysis_start_time']:
            cap_in.set(cv2.CAP_PROP_POS_MSEC, task_entry['analysis_start_time'].total_seconds() * 1e3)
            timer = timedelta(seconds=task_entry['analysis_start_time'].total_seconds())

    while cap_in.isOpened() and not (is_closed and is_closed()):
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
                    # device='cpu',
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
            corpus_infos = detector.output_corpus(output_dir, frame)

            if not corpus_infos:
                continue

            for corpus_info in corpus_infos:
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

                entry = {
                    'model_name': model['model_name'],
                    'model_version': model['model_version'],
                    'camera_type': camera['type'],
                    'camera_id': camera['camera_id'],
                    'video_type': 1 if is_online else 2,
                    'source': video_url,
                    'dest': corpus_info['dest'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'plate_no': corpus_info['plate_no'],
                    'locations': camera['matrix']  # TODO
                }

                append_corpus_signal.emit(entry)  # Update ui
                RESULT_DAO.insert_result(entry)  # Update sql
                # TODO for test. Need uncomment.
                # OBS_DAO.upload_file(f'{output_dir}/{entry['dest']}')  # Update obs

        set_umat and set_umat(plotted_frame)

        timer += timedelta(seconds=1 / fps)

        total_cost = time() - st0

        print(f'Total cost: {total_cost:.3f} s')
        print('---------------------')

        sleep(max(1 / fps - total_cost, 0))

    cap_in.release()

    # TODO for test. Need uncomment.
    update_task_status(task_entry['id'], 1 if end_as_designed else -1)
