from datetime import timedelta

from matplotlib import colors

from config import *
from db.camera import get_camera_by_camera_id
from db.group import get_group_by_group_id
from db.model import get_model_by_model_id
from db.result import insert_result
from db.task_offline import update_offline_task_status_by_id
from detectors import *
from utils import is_in_analysis, is_url


def run():
    crop_size: dict = GENERAL_CONFIG['crop']
    crop_top_y = None if crop_size['top_y'] == 'none' else int(crop_size['top_y'])
    crop_bottom_y = None if crop_size['bottom_y'] == 'none' else int(crop_size['bottom_y'])
    crop_left_x = None if crop_size['left_x'] == 'none' else int(crop_size['left_x'])
    crop_right_x = None if crop_size['right_x'] == 'none' else int(crop_size['right_x'])

    vehicle_det_model = YOLO(GENERAL_CONFIG['pre_trained'])
    cap_in = cv2.VideoCapture(GENERAL_CONFIG['source'])
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter(GENERAL_CONFIG['dest'], cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

    volume_detector = VolumeDetector(fps=fps, **VOLUME_CONFIG) if GENERAL_CONFIG['det_volume'] else None
    velocity_detector = VelocityDetector(fps=fps, **VELOCITY_CONFIG) if GENERAL_CONFIG['det_velocity'] else None
    polume_detector = PolumeDetector(**POLUME_CONFIG) if GENERAL_CONFIG['det_polume'] else None
    size_detector = SizeDetector(fps=fps, **SIZE_CONFIG) if GENERAL_CONFIG['det_size'] else None
    color_classifier = ColorClassifier(**COLOR_CONFIG) if GENERAL_CONFIG['clas_color'] else None
    pim_detector = PimDetector(fps=fps, **PIM_CONFIG) if GENERAL_CONFIG['det_pim'] else None
    parking_detector = ParkingDetector(fps=fps, **PARKING_CONFIG) if GENERAL_CONFIG['det_parking'] else None
    wrongway_detector = WrongwayDetector(fps=fps, **WRONGWAY_CONFIG) if GENERAL_CONFIG['det_wrongway'] else None
    lanechange_detector = LanechangeDetector(fps=fps, **LANECHANGE_CONFIG) if GENERAL_CONFIG['det_lanechange'] else None
    speeding_detector = SpeedingDetector(fps=fps, **SPEEDING_CONFIG) if GENERAL_CONFIG['det_speeding'] else None

    stats_height = 30
    subscript_height = 12
    px_per_scale = 30

    while cap_in.isOpened():
        st0 = time()

        ret, frame = cap_in.read()
        if not ret:
            break

        stats_line = 0
        subscript_line = 0

        frame = frame[crop_top_y: crop_bottom_y, crop_left_x: crop_right_x, :]

        st1 = time()
        result = vehicle_det_model.track(
            source=frame,
            imgsz=width,
            save=False,
            agnostic_nms=True,
            persist=True,
            verbose=False,
            device=0,
            # classes=[2, 5, 7]
            classes=[0, 3]
        )[0].cpu().numpy()
        print(f'Model cost: {time() - st1:.3f} ms')

        img = result.plot(conf=False, line_width=1)

        if GENERAL_CONFIG['det_jam']:
            stats_line += 1
            ret_jam = detect_jam(result, **JAM_CONFIG)
            cv2.putText(
                img,
                f'Jam: {ret_jam}',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                stats_height / px_per_scale,
                (0, 0, 255)
            )

        if GENERAL_CONFIG['det_queue']:
            stats_line += 1
            ret_queue = detect_queue(result, **QUEUE_CONFIG)
            cv2.polylines(img, np.array([QUEUE_CONFIG['det_zone']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                img,
                f'Queue: {ret_queue:.2f} m',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                stats_height / px_per_scale,
                (0, 255, 255)
            )

        if GENERAL_CONFIG['det_density']:
            stats_line += 1
            ret_density = detect_density(result, **DENSITY_CONFIG)
            cv2.polylines(img, np.array([DENSITY_CONFIG['det_zone']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                img,
                f'Density: {ret_density:.2f} cars per km',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                stats_height / px_per_scale,
                (0, 255, 255)
            )

        if GENERAL_CONFIG['det_size']:
            subscript_line += 1
            ret_size = size_detector.update(result)
            cv2.line(img, SIZE_CONFIG['vertices'][0], SIZE_CONFIG['vertices'][1], color=(255, 0, 0))
            for idx in ret_size:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                color = (200, 0, 0) if ret_size[idx] == 'small' else (
                    (0, 200, 0) if ret_size[idx] == 'medium' else (0, 0, 200)
                )
                cv2.putText(
                    img,
                    f'{ret_size[idx]}',
                    (int(xyxy[2]), int(xyxy[1]) + subscript_height * subscript_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    color
                )

        if GENERAL_CONFIG['clas_color']:
            subscript_line += 1
            ret_color = color_classifier.classify(result, frame)
            for idx in ret_color:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                color = tuple(map(lambda x: int(x * 255), colors.hex2color(colors.cnames[ret_color[idx]])[::-1]))
                cv2.putText(
                    img,
                    f'{ret_color[idx]}',
                    (int(xyxy[2]), int(xyxy[1]) + subscript_height * subscript_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    color
                )

        if GENERAL_CONFIG['det_volume']:
            stats_line += 1
            ret_volume = volume_detector.update(result, frame)
            cv2.line(img, VOLUME_CONFIG['det_line'][0], VOLUME_CONFIG['det_line'][1], color=(0, 255, 0), thickness=2)
            # cv2.polylines(img, np.array([VOLUME_CONFIG['det_zone']]), isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img,
                f'Volume: {ret_volume}',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                stats_height / px_per_scale,
                (0, 255, 0)
            )

        if GENERAL_CONFIG['det_velocity']:
            subscript_line += 1
            ret_velocity = velocity_detector.update(result, frame)
            cv2.polylines(img, np.array([VELOCITY_CONFIG['det_zone']]), isClosed=True, color=(255, 255, 0), thickness=2)
            for idx in ret_velocity:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                cv2.putText(
                    img,
                    f'{ret_velocity[idx]:.3f} km/h',
                    (int(xyxy[2]), int(xyxy[1] + subscript_height * subscript_line)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    (255, 255, 0)
                )

        if GENERAL_CONFIG['det_polume']:
            stats_line += 1
            ret_polume = polume_detector.update(result)
            # cv2.polylines(img, np.array([POLUME_CONFIG['vertices']]), isClosed=True, color=(0, 100, 255), thickness=2)
            cv2.line(img, SIZE_CONFIG['vertices'][0], SIZE_CONFIG['vertices'][1], color=(255, 0, 0))
            cv2.putText(
                img,
                f'Pedestrian volume: {ret_polume}',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                stats_height / px_per_scale,
                (0, 100, 255)
            )

        if GENERAL_CONFIG['det_pim']:
            ret_pim = pim_detector.update(result, frame)
            cv2.polylines(img, np.array([PIM_CONFIG['det_zone']]), isClosed=True, color=(0, 0, 255), thickness=2)
            for idx in ret_pim:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                tl = (int(xyxy[0]), int(xyxy[1]))
                br = (int(xyxy[2]), int(xyxy[3]))
                color = (0, 0, 200) if ret_pim[idx] else (0, 200, 0)
                cv2.rectangle(img, tl, br, color=color, thickness=2)

        if GENERAL_CONFIG['det_parking']:
            ret_parking = parking_detector.update(result, frame)
            cv2.polylines(img, np.array([PARKING_CONFIG['det_zone']]), isClosed=True, color=(0, 0, 255), thickness=2)
            for idx in ret_parking:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                state = ret_parking[idx]
                color = (0, 0, 255) if state == 'illegally parked' else (0, 255, 0)  # duplicated for demo
                text = state if state == 'illegally parked' else ''  # duplicated for demo
                cv2.putText(
                    img,
                    text,
                    (int(xyxy[0]), int(xyxy[3] + subscript_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    color
                )

        if GENERAL_CONFIG['det_wrongway']:
            subscript_line += 1
            ret_wrongway = wrongway_detector.update(result, frame)
            cv2.polylines(img, np.array([WRONGWAY_CONFIG['det_zone']]), isClosed=True, color=(0, 255, 255), thickness=2)
            for idx in ret_wrongway:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xywh = result.boxes.xywh[retrieve]
                xyxy = result.boxes.xyxy[retrieve]
                vector = ret_wrongway[idx]['vector']
                wrongway = ret_wrongway[idx]['wrongway']
                color = (0, 0, 255) if wrongway else (0, 255, 0)
                cv2.arrowedLine(
                    img,
                    (int(xywh[0]), int(xywh[1])),
                    (int(xywh[0] + 50 * vector[0]), int(xywh[1] + 50 * vector[1])),
                    color=color
                )
                cv2.putText(
                    img,
                    'wrong way' if wrongway else 'right way',
                    (int(xyxy[2]), int(xyxy[1] + subscript_height * subscript_line)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    color
                )

        if GENERAL_CONFIG['det_lanechange']:
            ret_lanechange = lanechange_detector.update(result, frame)
            cv2.polylines(img, np.array([LANECHANGE_CONFIG['det_zone']]), True, color=(127, 127, 0), thickness=2)
            for solid_line in LANECHANGE_CONFIG['solid_lines']:
                cv2.line(img, solid_line[0], solid_line[1], color=(255, 255, 255), thickness=2)
            for idx in ret_lanechange:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                lanechange = ret_lanechange[idx]
                cv2.putText(
                    img,
                    'changing' if lanechange else 'staying',
                    (int(xyxy[0]), int(xyxy[3] + subscript_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    (0, 0, 255) if lanechange else (0, 255, 0)
                )

        if GENERAL_CONFIG['det_speeding']:
            ret_speeding = speeding_detector.update(result, frame)
            cv2.polylines(img, np.array([SPEEDING_CONFIG['det_zone']]), isClosed=True, color=(255, 255, 0), thickness=2)
            for idx in ret_speeding:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                speeding = ret_speeding[idx]
                cv2.putText(
                    img,
                    'speeding' if speeding else 'not speeding',
                    (int(xyxy[0]), int(xyxy[3] + subscript_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    subscript_height / px_per_scale,
                    (0, 0, 255) if speeding else (0, 255, 0)
                )

        print(f'Total cost: {time() - st0:.3f} ms')
        print('---------------------')

        cap_out.write(img)
        cv2.imshow('Test', img)
        if cv2.waitKey(1) >= 0:
            break

    cap_in.release()
    cap_out.release()


def new_run_offline(
        task_entry: dict,
        output_dir: str,
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

    timer = timedelta(seconds=0)

    while cap_in.isOpened():
        st0 = time()

        ret, frame = cap_in.read()
        if not ret:
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
                    entry = [
                        model_entry['model_name'],
                        model_entry['model_version'],
                        camera_entry['type'],
                        camera_entry['camera_id'],
                        is_url(camera_entry['url']),
                        camera_entry['url'],
                        dest,
                        task_entry['analysis_start_time'],  # TODO
                        task_entry['analysis_end_time'],  # TODO
                        None,
                        []  # TODO
                    ]
                    insert_result(entry)

        print(f'Total cost: {time() - st0:.3f} ms')
        print('---------------------')

        timer += timedelta(seconds=1/fps)

        if plotted_frame is not None:
            cap_out.write(plotted_frame)
            cv2.imshow('Test', plotted_frame)

            if cv2.waitKey(1) >= 0:
                break

    cap_in.release()
    cap_out.release()

    update_offline_task_status_by_id(task_entry['id'], 1)
