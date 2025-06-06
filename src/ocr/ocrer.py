import copy
from typing import Any

import cv2
import numpy as np

from ocr.config import parse_args
from ocr.text_classifier import TextClassifier
from ocr.text_detector import TextDetector
from ocr.text_recognizer import TextRecognizer


class OCRer:
    def __init__(self, **kwargs):
        params = parse_args()
        params.__dict__.update(**kwargs)

        self.text_detector = TextDetector(params, **kwargs)
        self.text_recognizer = TextRecognizer(params, **kwargs)
        self.use_angle_cls = params.use_angle_cls
        self.drop_score = params.drop_score
        self.show_log = params.show_log

        if self.use_angle_cls:
            self.text_classifier = TextClassifier(params, **kwargs)

    @staticmethod
    def get_rotate_crop_image(img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height]
            ]
        )

        matrix = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            matrix,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]

        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)

        return dst_img

    @staticmethod
    def print_draw_crop_rec_res(img_crop_list, rec_res):
        for bno in range(len(img_crop_list)):
            cv2.imwrite(f'./output/img_crop_{bno}.jpg', img_crop_list[bno])
            print(bno, rec_res[bno])

    def ocr(
            self,
            img: str | cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray,
            imgsz: tuple = None  # (height, width)
    ):
        if isinstance(img, str):
            img = cv2.imread(img)

            if img is None:
                return [], []

        ori_im = img.copy()
        ori_sz = (img.shape[0], img.shape[1])  # (height, width)

        if imgsz is None:
            rsz_im = img.copy()

        else:
            rsz_im = cv2.resize(img, (imgsz[1], imgsz[0]))

        dt_boxes, elapse = self.text_detector(rsz_im)  # dt_boxes: ndarray = [[tl, tr, bt, bl], ...] [width, height]

        try:
            if imgsz is not None and dt_boxes.ndim == 3:
                dt_boxes[:, :, 0] *= ori_sz[1] / imgsz[1]
                dt_boxes[:, :, 1] *= ori_sz[0] / imgsz[0]

        except Exception as e:
            print(e)

        if self.show_log:
            print(f'dt_boxes num: {len(dt_boxes)}, elapse: {elapse}')

        if dt_boxes is None:
            return [], []

        img_crop_list = []

        dt_boxes = sort_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)

            if self.show_log:
                print(f'cls num: {len(img_crop_list)}, elapse: {elapse}')

        rec_res, elapse = self.text_recognizer(img_crop_list)

        if self.show_log:
            print(f'rec_res num: {len(rec_res)}, elapse: {elapse}')

        bboxes, filtered_rec_res = [], []

        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result

            if score >= self.drop_score:
                bboxes.append(box)
                filtered_rec_res.append(rec_result)

        return bboxes, filtered_rec_res


def sort_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array): detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))

    for i in range(dt_boxes.shape[0] - 1):
        if (
                abs(sorted_boxes[i + 1][0][1] - sorted_boxes[i][0][1]) < 10 and
                sorted_boxes[i + 1][0][0] < sorted_boxes[i][0][0]
        ):
            sorted_boxes[i], sorted_boxes[i + 1] = sorted_boxes[i + 1], sorted_boxes[i]

    return sorted_boxes
