import math
from collections import deque
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils import cal_euclidean_distance, cal_intersection_points, cal_homography_matrix, reproject, VehicleSize, \
    is_between, cal_intersection_ratio


def detect_jam(
        result: Results,
        cls_indices: list,
        n_threshold: int
) -> bool:
    count = 0

    for ele in result.boxes.cls:
        if ele in cls_indices:
            count += 1

            if count >= n_threshold:
                return True

    return False


def detect_motor_into_pavement(
        result: Results
) -> bool:
    # TODO
    return True


def detect_size(
        result: Results,
        vertices: list
) -> dict[int, int]:
    # TODO

    return {1: VehicleSize.LARGE}


def detect_queue(
        result: Results,
        cls_indices: list,
        vertices: list,
        lengths_m: list,
        angle: float,
        int_threshold: float
) -> float:
    radian = math.pi * angle / 180
    rv0 = (0., 0.)
    rv1 = (0., lengths_m[0])
    rv3 = (lengths_m[-1] * math.sin(radian), lengths_m[-1] * math.cos(radian))

    points = cal_intersection_points(rv1, rv3, lengths_m[1], lengths_m[2])

    if len(points) == 1:
        rv2 = points[0]

    else:
        d_a = cal_euclidean_distance(rv0, points[0])
        d_b = cal_euclidean_distance(rv0, points[1])
        rv2 = points[0] if d_a > d_b else points[1]

    homography_matrix = cal_homography_matrix(vertices, [rv0, rv1, rv2, rv3])

    polygon = Polygon(vertices)
    head_point = ()
    tail_point = ()

    for i, xyxy in enumerate(result.boxes.xyxy):
        if result.boxes.cls[i] not in cls_indices:
            continue

        bl = (xyxy[0], xyxy[3])
        br = (xyxy[2], xyxy[3])
        line = LineString([bl, br])

        if line.intersection(polygon).length / line.length >= int_threshold:
            head_point = br if not head_point else (head_point if head_point[1] <= br[1] else br)
            tail_point = br if not tail_point else (tail_point if tail_point[1] > br[1] else br)

    if head_point and tail_point:
        points = np.array([head_point, tail_point])
        repro_points = reproject(points, homography_matrix)
        distance = cal_euclidean_distance(repro_points[0], repro_points[1])
        return distance

    else:
        return 0


def detect_density(
        result: Results,
        cls_indices: list,
        vertices: list,
        int_threshold: float,
        length_m: float
) -> float:
    polygon = Polygon(vertices)
    count = 0

    for i, xyxy in enumerate(result.boxes.xyxy):
        if result.boxes.cls[i] not in cls_indices:
            continue

        line = LineString([(xyxy[0], xyxy[3]), (xyxy[2], xyxy[3])])

        if line.intersection(polygon).length / line.length >= int_threshold:
            count += 1

    if length_m == 0:
        return 0

    else:
        return count / length_m * 1e3


class ColorClassifier:
    def __init__(
            self,
            cls_indices: list,
            resize: list,
            min_size: list,
            weights='weights/resnet50.pth',
            cls_path='configs/color.yaml'
    ):
        self._res = {}
        self._cls_indices = cls_indices
        self._idx2cls: dict = yaml.safe_load(open(cls_path, 'r'))
        self._model = YOLO(weights)
        self._transform = transforms.Compose([
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self._min_size = min_size

    def classify(
            self,
            result: Results,
            img: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
    ) -> dict[int, str]:
        if result.boxes.id is None:
            return {}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ret = {}
        idxes = []
        box_imgs = []
        for i, idx in enumerate(result.boxes.id):
            x0, y0, w, h = result.boxes.xywh[i]
            if result.boxes.cls[i] not in self._cls_indices:
                continue

            if w < self._min_size[0] or h < self._min_size[1]:
                if self._res.get(idx):
                    ret[idx] = self._res[idx]
                continue

            xyxy = result.boxes.xyxy[i]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            box_img = img[y1: y2, x1: x2, :]
            box_img = self._transform(Image.fromarray(box_img))

            idxes.append(idx)
            box_imgs.append(box_img)

        preds = self._model(torch.stack(box_imgs), verbose=False, device=0) if box_imgs else {}

        for i, pred in enumerate(preds):
            label = self._idx2cls[pred.probs.top1]
            self._res[idxes[i]] = label
            ret[idxes[i]] = label

        return ret


class VolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            vertices: list
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(vertices)

    def update(
            self,
            result: Results
    ) -> int:
        for i, xywh in enumerate(result.boxes.xywh):
            if result.boxes.cls[i] not in self._cls_indices:
                continue

            centre = Point((xywh[0], xywh[1]))
            if self._polygon.contains(centre):
                self._id_set.add(result.boxes.id[i])

        return len(self._id_set)


class SpeedDetector:
    def __init__(
            self,
            cls_indices: list,
            vertices: list,
            lengths_m: list,
            angle: float,
            grain_size_second: float,
            fps: float
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(vertices)
        self._fps = fps
        self._buffer = deque(maxlen=max(round(grain_size_second * fps), 1))

        radian = math.pi * angle / 180
        rv0 = (0., 0.)
        rv1 = (0., lengths_m[0])
        rv3 = (lengths_m[-1] * math.sin(radian), lengths_m[-1] * math.cos(radian))

        points = cal_intersection_points(rv1, rv3, lengths_m[1], lengths_m[2])

        if len(points) == 1:
            rv2 = points[0]

        else:
            d_a = cal_euclidean_distance(rv0, points[0])
            d_b = cal_euclidean_distance(rv0, points[1])
            rv2 = points[0] if d_a > d_b else points[1]

        self._homography_matrix = cal_homography_matrix(vertices, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results
    ) -> dict[int, float]:
        if result.boxes.id is None:
            return {}

        if len(self._buffer) == 0:
            self._buffer.append(result)
            return {idx: 0 for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            centre = Point((result.boxes.xywh[i][0], result.boxes.xywh[i][1]))
            if result.boxes.cls[i] not in self._cls_indices or not self._polygon.contains(centre):
                continue

            speed = 0

            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    old_x, old_y, *_ = buf.boxes.xywh[list_idx]
                    now_x, now_y, *_ = result.boxes.xywh[i]

                    points = np.array([[now_x, now_y], [old_x, old_y]])
                    repro_points = reproject(points, self._homography_matrix)
                    distance = cal_euclidean_distance(repro_points[0], repro_points[1])
                    speed = distance / ((len(self._buffer) - j) / self._fps) * 3.6
                    break

            ret[idx] = speed

        self._buffer.append(result)

        return ret


class PolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            iou_threshold: float
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._iou_threshold = iou_threshold

    def update(
            self,
            result: Results
    ) -> int:
        vehicle_retrieves = np.where(result.boxes.cls == self._cls_indices[1])[0]

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self._cls_indices[0:]:
                continue

            walking = True

            for retrieve in vehicle_retrieves:
                iou = cal_intersection_ratio(xyxy, result.boxes.xyxy[retrieve])

                if iou > self._iou_threshold:
                    walking = False
                    break

            if walking:
                self._id_set.add(result.boxes.id[i])

        return len(self._id_set)


def detect_pim(
        result: Results,
        cls_indices: list,
        vertices: list,
        iou_threshold=0.6
) -> dict[int, bool]:
    ret = {}

    vehicle_retrieves = np.where(result.boxes.cls == cls_indices[1])[0]

    for i, xyxy in enumerate(result.boxes.xyxy):
        if result.boxes.cls[i] not in cls_indices[0:]:
            continue

        walking = True

        for retrieve in vehicle_retrieves:
            iou = cal_intersection_ratio(xyxy, result.boxes.xyxy[retrieve])

            if iou > iou_threshold:
                walking = False
                break

        if walking:
            upper = LineString([vertices[0], vertices[3]])
            lower = LineString([vertices[1], vertices[2]])
            line = LineString([(xyxy[0], xyxy[3]), (xyxy[2], xyxy[3])])
            ret[result.boxes.id[i]] = not (line.intersects(lower) or is_between(line, lower, upper))

    return ret
