import argparse

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', type=str, default='D:/xxs-signs/vehicle-detection/resources/traffic.mp4')
    parser.add_argument('-d', type=str, default='D:/xxs-signs/vehicle-detection/resources/traffic.png')
    parser.add_argument('-p', type=int, default=1 * 1e3)

    args = parser.parse_args()

    cap_in = cv2.VideoCapture(args.s)
    cap_in.set(cv2.CAP_PROP_POS_MSEC, args.p)
    ret, frame = cap_in.read()
    cv2.imwrite(args.d, frame)
    cap_in.release()
