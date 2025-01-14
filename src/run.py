import cv2
from ultralytics import YOLO

from config import ARG_CONFIG
from detect import detect_jam, detect_motor_into_pavement


def run():
    det_jam: bool = ARG_CONFIG['det_jam']
    det_motor_into_pavement: bool = ARG_CONFIG['det_motor_into_pavement']
    det_volume: bool = ARG_CONFIG['det_volume']

    if det_jam or det_motor_into_pavement or det_volume:
        vehicle_det_model = YOLO('yolo11s.pt')
        cap_in = cv2.VideoCapture(ARG_CONFIG['source'])

        while True:
            ret, frame = cap_in.read()

            if not ret:
                break

            result = vehicle_det_model.predict(frame, imgsz=frame.shape[1], save=False, device=0)[0].cpu().numpy()
            # results = vehicle_det_model.track(frame, imgsz=frame.shape[1], device=0, save=False, persist=True)

            if det_jam:
                ret_jam = detect_jam(result)
                print(ret_jam)

            if det_motor_into_pavement:
                ret_motor_into_pavement = detect_motor_into_pavement(result)
                print(ret_motor_into_pavement)

            break

            # img = result.plot(conf=False, labels=False, line_width=1)
            # cv2.imshow('test', img)
            # if cv2.waitKey(1) >= 0:
            #     break

        cap_in.release()


if __name__ == '__main__':
    run()
