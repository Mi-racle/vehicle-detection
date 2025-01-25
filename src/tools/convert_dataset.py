import os

import cv2
import numpy as np


def seg2clas(src: str, dst: str, cls2size_dict: dict, size2name_dict: dict):
    sets = ['test', 'train', 'valid']

    for _set in sets:
        src_set_dir = os.path.join(src, _set)
        dst_set_dir = os.path.join(dst, 'val' if _set == 'valid' else _set)

        if not os.path.exists(src_set_dir):
            blk_img = np.zeros((224, 224, 3), dtype=np.uint8)

            for value in size2name_dict.values():
                dst_img_dir = os.path.join(dst_set_dir, value)
                os.makedirs(dst_img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dst_img_dir, '0.jpg'), blk_img)

            continue

        os.makedirs(dst_set_dir, exist_ok=True)

        for img_filename in os.listdir(os.path.join(src_set_dir, 'images')):
            lbl_filename = os.path.splitext(img_filename)[0] + '.txt'

            src_img_path = os.path.join(src_set_dir, 'images', img_filename)
            src_lbl_path = os.path.join(src_set_dir, 'labels', lbl_filename)

            src_img = cv2.imread(src_img_path)
            if src_img is None:
                continue

            height, width, *_ = src_img.shape

            with open(src_lbl_path, 'r') as fin:
                lines = fin.readlines()

            for i, line in enumerate(lines):
                splits = line.split()

                size = cls2size_dict[int(splits[0])]
                if size == -1:
                    continue

                dst_img_dir = os.path.join(dst_set_dir, size2name_dict[size])
                os.makedirs(dst_img_dir, exist_ok=True)

                dst_filename = f'{str(i)}{img_filename[:7]}.jpg'
                dst_img_path = os.path.join(dst_img_dir, dst_filename)
                xs = list(map(float, splits[1::2]))
                ys = list(map(float, splits[2::2]))
                x0, y0 = int(min(xs) * width), int(min(ys) * height)
                x1, y1 = int(max(xs) * width), int(max(ys) * height)

                if (x1 - x0) * (y1 - y0) < 96 ** 2:
                    continue

                box_img = src_img[y0: y1, x0: x1, :]

                cv2.imwrite(dst_img_path, box_img)


def det2clas(src: str, dst: str, cls2size_dict: dict, size2name_dict: dict):
    sets = ['test', 'train', 'valid']

    for _set in sets:
        src_set_dir = os.path.join(src, _set)
        dst_set_dir = os.path.join(dst, 'val' if _set == 'valid' else _set)

        if not os.path.exists(src_set_dir):
            blk_img = np.zeros((224, 224, 3), dtype=np.uint8)

            for value in size2name_dict.values():
                dst_img_dir = os.path.join(dst_set_dir, value)
                os.makedirs(dst_img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dst_img_dir, '0.jpg'), blk_img)

            continue

        os.makedirs(dst_set_dir, exist_ok=True)

        for img_filename in os.listdir(os.path.join(src_set_dir, 'images')):
            lbl_filename = os.path.splitext(img_filename)[0] + '.txt'

            src_img_path = os.path.join(src_set_dir, 'images', img_filename)
            src_lbl_path = os.path.join(src_set_dir, 'labels', lbl_filename)

            src_img = cv2.imread(src_img_path)
            if src_img is None:
                continue

            height, width, *_ = src_img.shape

            with open(src_lbl_path, 'r') as fin:
                lines = fin.readlines()

            for i, line in enumerate(lines):
                splits = line.split()

                size = cls2size_dict[int(splits[0])]
                if size == -1:
                    continue

                dst_img_dir = os.path.join(dst_set_dir, size2name_dict[size])
                os.makedirs(dst_img_dir, exist_ok=True)

                dst_filename = f'{str(i)}{img_filename[:7]}.jpg'
                dst_img_path = os.path.join(dst_img_dir, dst_filename)
                xywh = list(map(float, splits[1:]))
                x0, y0 = int((xywh[0] - xywh[2] / 2) * width), int((xywh[1] - xywh[3] / 2) * height)
                x1, y1 = int((xywh[0] + xywh[2] / 2) * width), int((xywh[1] + xywh[3] / 2) * height)

                if (x1 - x0) * (y1 - y0) < 96 ** 2:
                    continue

                box_img = src_img[y0: y1, x0: x1, :]

                cv2.imwrite(dst_img_path, box_img)


if __name__ == '__main__':
    dst_path = '../../resources/simplified-car-size-dataset'
    size2name = {0: 'small', 1: 'medium', 2: 'large'}  # value: -1:to exclude, 0-small, 1-medium, 2-large

    src_path = '../../resources/car-size-dataset2'
    # car_size_dataset2_cls2size = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: -1, 7: 2, 8: 2, 9: 0, 10: -1, 11: 1, 12: 1,
    #                               13: 0, 14: 1, 15: 2, 16: 2, 17: 1, 18: 2, 19: 2, 20: 1}
    car_size_dataset2_cls2size = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: -1, 6: -1, 7: 2, 8: 2, 9: -1, 10: -1, 11: -1, 12: -1,
                                  13: -1, 14: -1, 15: -1, 16: 2, 17: 1, 18: 2, 19: 2, 20: -1}
    seg2clas(src_path, dst_path, car_size_dataset2_cls2size, size2name)

    # src_path = '../../resources/car-size-dataset'
    # car_size_dataset_cls2size = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1}
    # det2clas(src_path, dst_path, car_size_dataset_cls2size, size2name)
