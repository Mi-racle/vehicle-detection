import logging
import os
from datetime import timedelta, datetime
from time import time, sleep
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PyQt6.QtCore import pyqtSignal
from ultralytics import YOLO

from db import GROUP_DAO, TASK_ONLINE_DAO, TASK_OFFLINE_DAO, MODEL_DAO, SYSCON_DAO
from detectors import ParkingDetector, WrongwayDetector, LanechangeDetector, SpeedingDetector, VelocityDetector, \
    PimDetector, SectionDetector, VolumeDetector, DensityDetector, QueueDetector, JamDetector, PlateDetector, \
    TriangleDetector, SizeDetector, ObjectDetector
from ocr.ocrer import OCRer
from utils import get_file_by_substr, in_analysis, filter_kwargs, download_file, generate_hash


def detect(
        task_entry: dict,
        output_dir='runs',
        weight_dir='weights',
        det_width=480,
        online=False,
        use_gpu=False,
        warmup=True,
        is_closed: Callable[[], bool] | None = None,
        append_corpus_signal: pyqtSignal | None = None,
        set_umat: Callable[[cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray], None] | None = None,
):
    if online:
        update_task_status = TASK_ONLINE_DAO.update_online_task_status_by_id
    else:
        update_task_status = TASK_OFFLINE_DAO.update_offline_task_status_by_id

    cap_in = cv2.VideoCapture(task_entry['url'])
    fps = (cap_in.get(cv2.CAP_PROP_FPS) or 25)

    det_models: dict[str, dict[str, Any]] = {}
    detectors = {}  # {'group_id': Detector}
    dets_args = {}  # {'group_id': det_args}
    model_entries = {}  # {'group_id': model_entry}

    for group_id in task_entry['group_id']:
        group_entry = GROUP_DAO.get_group_by_group_id(group_id)

        if not group_entry:
            continue

        model_entry = MODEL_DAO.get_model_by_model_id(group_entry['model_id'])
        det_args = group_entry['args']

        model_entries[group_id] = model_entry
        dets_args[group_id] = det_args

        if model_entry['file_path'] not in det_models:
            mdl = None
            model_paths = model_entry['file_path'].split(',')

            if 'yolo' in model_entry['file_path']:
                model_path = model_paths[0]
                model_url = f'{weight_dir}/{os.path.basename(model_path)}'
                url_prefix = SYSCON_DAO.get_url_prefix()

                if not os.path.exists(model_url):
                    if download_file(url_prefix + model_path, model_url):
                        logging.info(f'Model {model_path} successfully downloaded to {model_url}')
                    else:
                        logging.warning(f'Failed to download model {model_path}')
                        update_task_status(task_entry['id'], -1)
                        cap_in.release()
                        return

                mdl = YOLO(model_url)

            elif 'ocr' in model_entry['file_path']:
                ocr_dir = 'ocr_' + generate_hash(model_entry['file_path'], num=20)
                dir_path = os.path.join(weight_dir, ocr_dir)
                model_urls = [os.path.join(dir_path, os.path.basename(model_path)) for model_path in model_paths]
                url_prefix = SYSCON_DAO.get_url_prefix()

                for model_i, model_path in enumerate(model_paths):
                    model_url = model_urls[model_i]

                    if os.path.exists(model_url):
                        continue

                    if download_file(url_prefix + model_path, model_url):
                        logging.info(f'Model {model_path} successfully downloaded to {model_url}')
                    else:
                        logging.warning(f'Failed to download model {model_path}')
                        update_task_status(task_entry['id'], -1)
                        cap_in.release()
                        return

                det_file = get_file_by_substr(dir_path, 'det')
                rec_file = get_file_by_substr(dir_path, 'rec')

                if det_file and rec_file:
                    mdl = OCRer(
                        det_model_path=os.path.join(dir_path, det_file),
                        rec_model_path=os.path.join(dir_path, rec_file),
                        use_gpu=use_gpu & torch.cuda.is_available()
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
        elif model_entry['model_type'] == 1:
            detectors[group_id] = ParkingDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 2:
            detectors[group_id] = WrongwayDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 3:
            detectors[group_id] = SpeedingDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 4:
            detectors[group_id] = PimDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 5:
            detectors[group_id] = LanechangeDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 6:
            detectors[group_id] = VelocityDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 7:
            detectors[group_id] = SectionDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 8:
            detectors[group_id] = VolumeDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 9:
            detectors[group_id] = DensityDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 10:
            detectors[group_id] = QueueDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 11:
            detectors[group_id] = JamDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 12:
            detectors[group_id] = PlateDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 13:
            detectors[group_id] = TriangleDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 14:
            detectors[group_id] = SizeDetector(fps=fps, **det_args)
        elif model_entry['model_type'] == 15:
            detectors[group_id] = ObjectDetector(fps=fps, **det_args)
        else:
            logging.info('Unknown model found')
            update_task_status(task_entry['id'], -1)
            cap_in.release()
            return

        logging.info(f'Detector (group id {group_id}) successfully inited')

    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_as_designed = False
    timer = timedelta(seconds=1 / fps)

    if online:
        now = datetime.now()
        timer = timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)

    elif task_entry['analysis_start_time']:
        cap_in.set(cv2.CAP_PROP_POS_MSEC, task_entry['analysis_start_time'].total_seconds() * 1e3)
        timer = timedelta(seconds=task_entry['analysis_start_time'].total_seconds())

    logging.info(f'Detection of {task_entry['url']} started')

    last_results = {}
    accrued_deviation = 0
    cap_open_patience = 3
    cap_read_patience = 3

    while True:
        if not cap_in.isOpened():
            cap_open_patience -= 1
            logging.info(f'Open patience - 1')

            if cap_open_patience < 0:
                logging.info(f'Beyond open patience. Cap is closed. Break')
                end_as_designed = True
                break

            else:
                logging.info(f'Within open patience. Continue')
                sleep(1 / fps)
                continue

        if is_closed and is_closed():
            logging.info('Process is closed')
            break

        st0 = time()

        logging.info('Try to read frame')
        ret, frame = cap_in.read()
        logging.info('Frame obtained')

        if not ret:
            cap_read_patience -= 1
            logging.info(f'Read patience - 1')

            if cap_read_patience < 0:
                logging.info(f'Beyond read patience. Break')
                end_as_designed = True
                break

            else:
                logging.info(f'Within read patience. Continue')
                sleep(1 / fps)
                continue

        st1 = time()

        plotted_frame = frame.copy()

        if not in_analysis(timer, task_entry['analysis_start_time'], task_entry['analysis_end_time']):
            logging.info('Video not in analysis')
            end_as_designed = True
            break

        results = {}  # {group_id: result}

        if False and online and accrued_deviation >= 1 / fps:  # TODO skip or not
            accrued_deviation -= 1 / fps
            results = last_results

            logging.info(f'Frame at {timer} skipped inference')

        else:
            for values in det_models.values():
                det_model = values['model']
                imgsz = (min(det_width * height // width, height), min(det_width, width))

                logging.info('Inference begins')

                if isinstance(det_model, YOLO):
                    result = det_model.track(
                        source=frame,
                        imgsz=imgsz,
                        save=False,
                        agnostic_nms=True,
                        persist=True,
                        verbose=False,
                        device=0 if use_gpu & torch.cuda.is_available() else 'cpu',
                        classes=values['classes']
                    )[0].cpu().numpy()
                elif isinstance(det_model, OCRer):
                    result = det_model.ocr(
                        img=frame,
                        imgsz=imgsz
                    )
                else:
                    result = None
                    logging.error('Result is None')

                logging.info('Inference ends')

                for group in values['groups']:
                    results[group] = result

        logging.info(f'Model cost: {time() - st1:.3f} s')

        if warmup:
            warmup = False
            logging.info(f'Warmup finished')
            continue

        last_results = results

        stats_line = 1
        subscript_line = 1
        velocities: dict[float, float] | None = None

        for group in detectors:  # 'group' is group id
            if group not in results:
                continue

            detector = detectors[group]
            model = model_entries[group]
            dargs = dets_args[group]

            kwargs = filter_kwargs(detector.update, {'result': results[group], 'velocities': velocities})
            ret = detector.update(**kwargs)
            if ret is not None:
                velocities = ret

            plotted_frame = detector.plot(plotted_frame, stats_line, subscript_line)
            logging.info('Detector.plot done')
            stats_line, subscript_line = detector.update_line(stats_line, subscript_line)
            logging.info('Detector.update_line done')
            corpus_infos = detector.output_corpus(output_dir, frame)
            logging.info('Detector.output_corpus done')

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

                start_time = timedelta(seconds=round(start_time.total_seconds()))
                end_time = timedelta(seconds=round(end_time.total_seconds()))

                if online and ('execute_date' in task_entry):
                    start_time = datetime.combine(task_entry['execute_date'], datetime.min.time()) + start_time
                    end_time = datetime.combine(task_entry['execute_date'], datetime.min.time()) + end_time

                entry = {
                    'model_name': model['model_name'],
                    'model_version': model['model_version'],
                    'camera_type': task_entry['camera_type'],
                    'camera_id': task_entry['camera_id'],
                    'video_type': 1 if online else 2,
                    'source': task_entry.get('download_url') or task_entry['url'],
                    'dest': corpus_info['dest'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'plate_no': corpus_info['plate_no'],
                    'locations': task_entry['matrix']  # TODO
                }

                append_corpus_signal.emit(entry, task_entry['description'])  # Update ui, sql and obs

        set_umat and set_umat(plotted_frame)

        timer += timedelta(seconds=1 / fps)

        total_cost = time() - st0

        logging.info(f'Total cost: {total_cost:.3f} s')
        logging.info('---------------------')

        if total_cost >= 1 / fps:
            accrued_deviation += total_cost - 1 / fps

        else:
            if accrued_deviation > 0:
                accrued_deviation -= 1 / fps - total_cost

            else:
                sleep(1 / fps - total_cost)

    logging.info(f'Detection of {task_entry['url']} ended')

    cap_in.release()

    update_task_status(task_entry['id'], 1 if end_as_designed else -1)
