import cv2
import numpy as np
from ultralytics import YOLO

from config import GENERAL_CONFIG, QUEUE_CONFIG, JAM_CONFIG, DENSITY_CONFIG, VOLUME_CONFIG, SPEED_CONFIG, MIP_CONFIG, \
    SIZE_CONFIG
from detect import detect_jam, detect_queue, detect_density, VolumeDetector, SpeedDetector, detect_size


def run():
    crop_size: dict = GENERAL_CONFIG['crop']
    crop_top_y = None if crop_size['top_y'] == 'none' else int(crop_size['top_y'])
    crop_bottom_y = None if crop_size['bottom_y'] == 'none' else int(crop_size['bottom_y'])
    crop_left_x = None if crop_size['left_x'] == 'none' else int(crop_size['left_x'])
    crop_right_x = None if crop_size['right_x'] == 'none' else int(crop_size['right_x'])

    vehicle_det_model = YOLO('yolo11s.pt')
    cap_out = None
    cap_in = cv2.VideoCapture(GENERAL_CONFIG['source'])
    fps = cap_in.get(cv2.CAP_PROP_FPS)

    volume_detector = VolumeDetector(**VOLUME_CONFIG) if GENERAL_CONFIG['det_volume'] else None
    speed_detector = SpeedDetector(fps=fps, **SPEED_CONFIG) if GENERAL_CONFIG['det_speed'] else None

    while cap_in.isOpened():
        ret, frame = cap_in.read()

        if not ret:
            break

        n_line = 0
        frame = frame[crop_top_y: crop_bottom_y, crop_left_x: crop_right_x, :]

        if not cap_out:
            cap_out = cv2.VideoWriter(f'demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

        result = vehicle_det_model.track(
            source=frame,
            imgsz=frame.shape[1],
            save=False,
            agnostic_nms=True,
            persist=True,
            device=0
        )[0].cpu().numpy()
        img = result.plot(conf=False, line_width=1)

        if GENERAL_CONFIG['det_jam']:
            n_line += 1
            ret_jam = detect_jam(result, **JAM_CONFIG)
            cv2.putText(img, f'Jam: {ret_jam}', (0, 30 * n_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        if GENERAL_CONFIG['det_size']:
            ret_size = detect_size(result, **SIZE_CONFIG)
            cv2.line(img, SIZE_CONFIG['vertices'][0], SIZE_CONFIG['vertices'][1], (255, 255, 0), thickness=2)

        if GENERAL_CONFIG['det_queue']:
            n_line += 1
            ret_queue = detect_queue(result, **QUEUE_CONFIG)
            cv2.polylines(img, np.array([QUEUE_CONFIG['vertices']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(img, f'Queue: {ret_queue:.2f} m', (0, 30 * n_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        if GENERAL_CONFIG['det_density']:
            n_line += 1
            ret_density = detect_density(result, **DENSITY_CONFIG)
            cv2.polylines(img, np.array([DENSITY_CONFIG['vertices']]), isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(img, f'Density: {ret_density:.2f} cars per km', (0, 30 * n_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

        if GENERAL_CONFIG['det_volume']:
            n_line += 1
            ret_volume = volume_detector.update(result)
            cv2.polylines(img, np.array([VOLUME_CONFIG['vertices']]), isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img, f'Volume: {ret_volume}', (0, 30 * n_line), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        if GENERAL_CONFIG['det_speed']:
            ret_speed = speed_detector.update(result)
            cv2.polylines(img, np.array([SPEED_CONFIG['vertices']]), isClosed=True, color=(255, 0, 0), thickness=2)
            for idx in ret_speed:
                retrieve = np.where(result.boxes.id == idx)[0][0]
                cv2.putText(img, f'{ret_speed[idx]:.3f} km/h', (int(result.boxes.xyxy[retrieve][2]), int(result.boxes.xyxy[retrieve][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

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
