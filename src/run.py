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
    color_classifier = ColorClassifier(**COLOR_CONFIG) if GENERAL_CONFIG['clas_color'] else None

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
            cap_out = cv2.VideoWriter(f'demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

        st1 = time()
        result = vehicle_det_model.track(
            source=frame,
            imgsz=frame.shape[1],
            save=False,
            agnostic_nms=True,
            persist=True,
            verbose=False,
            device=0,
        )[0].cpu().numpy()
        print(f'Model cost: {time() - st1:.3f} ms')

        img = result.plot(conf=False, line_width=1)

        if GENERAL_CONFIG['det_jam']:
            stats_line += 1
            ret_jam = detect_jam(result, **JAM_CONFIG)
            cv2.putText(img, f'Jam: {ret_jam}', (0, stats_height * stats_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        if GENERAL_CONFIG['det_size']:
            # TODO
            ret_size = detect_size(result, **SIZE_CONFIG)
            cv2.line(img, SIZE_CONFIG['vertices'][0], SIZE_CONFIG['vertices'][1], (255, 255, 0), thickness=2)

        if GENERAL_CONFIG['det_queue']:
            stats_line += 1
            ret_queue = detect_queue(result, **QUEUE_CONFIG)
            cv2.polylines(img, np.array([QUEUE_CONFIG['vertices']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(img, f'Queue: {ret_queue:.2f} m', (0, stats_height * stats_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        if GENERAL_CONFIG['det_density']:
            stats_line += 1
            ret_density = detect_density(result, **DENSITY_CONFIG)
            cv2.polylines(img, np.array([DENSITY_CONFIG['vertices']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(img, f'Density: {ret_density:.2f} cars per km', (0, stats_height * stats_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        if GENERAL_CONFIG['clas_color']:
            subscript_line += 1
            ret_color = color_classifier.classify(result, frame)
            for idx in ret_color:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                color = tuple(map(lambda x: int(x * 255), colors.hex2color(colors.cnames[ret_color[idx]])[::-1]))
                cv2.putText(img, f'{ret_color[idx]}', (int(result.boxes.xyxy[retrieve][2]), int(result.boxes.xyxy[retrieve][1]) + subscript_height * subscript_line), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)

        if GENERAL_CONFIG['det_volume']:
            stats_line += 1
            ret_volume = volume_detector.update(result)
            cv2.polylines(img, np.array([VOLUME_CONFIG['vertices']]), isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img, f'Volume: {ret_volume}', (0, stats_height * stats_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        if GENERAL_CONFIG['det_speed']:
            subscript_line += 1
            ret_speed = speed_detector.update(result)
            cv2.polylines(img, np.array([SPEED_CONFIG['vertices']]), isClosed=True, color=(255, 0, 0), thickness=2)
            for idx in ret_speed:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                cv2.putText(img, f'{ret_speed[idx]:.3f} km/h', (int(result.boxes.xyxy[retrieve][2:0:-1]), int(result.boxes.xyxy[retrieve][1]) + subscript_height * subscript_line), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

        if GENERAL_CONFIG['det_polume']:
            stats_line += 1
            ret_polume = polume_detector.update(result)
            # cv2.polylines(img, np.array([POLUME_CONFIG['vertices']]), isClosed=True, color=(0, 100, 255), thickness=2)
            cv2.putText(img, f'Pedestrian volume: {ret_polume}', (0, stats_height * stats_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255))

        if GENERAL_CONFIG['det_pim']:
            ret_pim = detect_pim(result, **PIM_CONFIG)
            # cv2.polylines(img, np.array([PIM_CONFIG['vertices']]), isClosed=True, color=(0, 200, 0), thickness=2)
            for idx in ret_pim:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                tl = (int(result.boxes.xyxy[retrieve][0]), int(result.boxes.xyxy[retrieve][1]))
                br = (int(result.boxes.xyxy[retrieve][2]), int(result.boxes.xyxy[retrieve][3]))
                color = (0, 0, 200) if ret_pim[idx] else (0, 200, 0)
                cv2.rectangle(img, tl, br, color=color, thickness=2)

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
