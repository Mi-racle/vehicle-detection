import argparse

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', type=str, default='D:/xxs-signs/vehicle-detection/resources/hangzhou.mp4')
    parser.add_argument('-d', type=str, default='D:/xxs-signs/vehicle-detection/resources/image.png')

    args = parser.parse_args()

    cap_in = cv2.VideoCapture(args.s)
    ret, frame = cap_in.read()
    cv2.imwrite(args.d, frame)
    cap_in.release()
