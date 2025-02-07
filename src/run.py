from time import time

from matplotlib import colors

from config import *
from detect import *


def run():
    crop_size: dict = GENERAL_CONFIG['crop']
    crop_top_y = None if crop_size['top_y'] == 'none' else int(crop_size['top_y'])
    crop_bottom_y = None if crop_size['bottom_y'] == 'none' else int(crop_size['bottom_y'])
    crop_left_x = None if crop_size['left_x'] == 'none' else int(crop_size['left_x'])
    crop_right_x = None if crop_size['right_x'] == 'none' else int(crop_size['right_x'])

    vehicle_det_model = YOLO(GENERAL_CONFIG['pre_trained'])
    cap_out = None
    cap_in = cv2.VideoCapture(GENERAL_CONFIG['source'])
    fps = cap_in.get(cv2.CAP_PROP_FPS)

    volume_detector = VolumeDetector(**VOLUME_CONFIG) if GENERAL_CONFIG['det_volume'] else None
    speed_detector = SpeedDetector(fps=fps, **SPEED_CONFIG) if GENERAL_CONFIG['det_speed'] else None
    polume_detector = PolumeDetector(**POLUME_CONFIG) if GENERAL_CONFIG['det_polume'] else None
    size_detector = SizeDetector(fps=fps, **SIZE_CONFIG) if GENERAL_CONFIG['det_size'] else None
    color_classifier = ColorClassifier(**COLOR_CONFIG) if GENERAL_CONFIG['clas_color'] else None
    parking_detector = ParkingDetector(fps=fps, **PARKING_CONFIG) if GENERAL_CONFIG['det_parking'] else None
    wrongway_detector = WrongwayDetector(fps=fps, **WRONGWAY_CONFIG) if GENERAL_CONFIG['det_wrongway'] else None
    lanechange_detector = LanechangeDetector(fps=fps, **LANECHANGE_CONFIG) if GENERAL_CONFIG['det_lanechange'] else None

    while cap_in.isOpened():
        st0 = time()

        ret, frame = cap_in.read()
        if not ret:
            break

        stats_line = 0
        stats_height = 30
        subscript_line = 0
        subscript_height = 12

        frame = frame[crop_top_y: crop_bottom_y, crop_left_x: crop_right_x, :]

        if not cap_out:
            cap_out = cv2.VideoWriter(
                f'demo.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame.shape[1], frame.shape[0])
            )

        st1 = time()
        result = vehicle_det_model.track(
            source=frame,
            imgsz=frame.shape[1],
            save=False,
            agnostic_nms=True,
            persist=True,
            verbose=False,
            device=0,
            classes=[2, 5, 7]
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
                1,
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
                1,
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
                1,
                (0, 255, 255)
            )

        if GENERAL_CONFIG['det_size']:
            subscript_line += 1
            ret_size = size_detector.update(result)
            # cv2.line(img, SIZE_CONFIG['vertices'][0], SIZE_CONFIG['vertices'][1], color=(255, 0, 0))
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
                    0.4,
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
                    0.4,
                    color
                )

        if GENERAL_CONFIG['det_volume']:
            stats_line += 1
            ret_volume = volume_detector.update(result)
            cv2.polylines(img, np.array([VOLUME_CONFIG['det_zone']]), isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img,
                f'Volume: {ret_volume}',
                (0, stats_height * stats_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0)
            )

        if GENERAL_CONFIG['det_speed']:
            subscript_line += 1
            ret_speed = speed_detector.update(result)
            cv2.polylines(img, np.array([SPEED_CONFIG['det_zone']]), isClosed=True, color=(255, 255, 0), thickness=2)
            for idx in ret_speed:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                cv2.putText(
                    img,
                    f'{ret_speed[idx]:.3f} km/h',
                    (int(xyxy[2]), int(xyxy[1] + subscript_height * subscript_line)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
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
                1,
                (0, 100, 255)
            )

        if GENERAL_CONFIG['det_pim']:
            ret_pim = detect_pim(result, **PIM_CONFIG)
            # cv2.polylines(img, np.array([PIM_CONFIG['det_zone']]), isClosed=True, color=(0, 200, 0), thickness=2)
            for idx in ret_pim:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                xyxy = result.boxes.xyxy[retrieve]
                tl = (int(xyxy[0]), int(xyxy[1]))
                br = (int(xyxy[2]), int(xyxy[3]))
                color = (0, 0, 200) if ret_pim[idx] else (0, 200, 0)
                cv2.rectangle(img, tl, br, color=color, thickness=2)

        if GENERAL_CONFIG['det_parking']:
            ret_parking = parking_detector.update(result)
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
                    0.4,
                    color
                )

        if GENERAL_CONFIG['det_wrongway']:
            subscript_line += 1
            ret_wrongway = wrongway_detector.update(result)
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
                    0.4,
                    color
                )

        if GENERAL_CONFIG['det_lanechange']:
            ret_lanechange = lanechange_detector.update(result)
            cv2.polylines(img, np.array([LANECHANGE_CONFIG['det_zone']]), isClosed=True, color=(127, 127, 0), thickness=2)
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
                    0.4,
                    (0, 0, 255) if lanechange else (0, 255, 0)
                )

        print(f'Total cost: {time() - st0:.3f} ms')
        print('---------------------')

        # cv2.imwrite('re.png', img)
        # break

        cap_out.write(img)
        cv2.imshow('test', img)
        if cv2.waitKey(1) >= 0:
            break

    cap_in.release()
    cap_out.release()


if __name__ == '__main__':
    run()
