from time import sleep

import cv2

if __name__ == '__main__':
    ddd = {'231': 33}
    cap_in = cv2.VideoCapture('../resources/traffic.mp4')
    while True:
        ret, frame = cap_in.read()
        if not ret:
            break

        cv2.imshow('test', frame)
        cv2.waitKey(1)
        sleep(1 / 12)
