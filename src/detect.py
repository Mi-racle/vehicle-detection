import math
from collections import deque
from typing import Any

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

from utils import cal_euclidean_distance, cal_intersection_points, cal_homography_matrix, reproject, is_between, \
    cal_intersection_ratio


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


def detect_queue(
        result: Results,
        cls_indices: list,
        det_zone: list,
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

    homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    polygon = Polygon(det_zone)
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
        det_zone: list,
        int_threshold: float,
        length_m: float
) -> float:
    polygon = Polygon(det_zone)
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


class SizeDetector:
    def __init__(
            self,
            cls_indices: list,
            vertices: list,
            thresholds: list,
            delta_second: float,
            fps: float
    ):
        self._res = {}
        self._cls_indices = cls_indices
        self._line = LineString(vertices)
        self._thresholds = thresholds
        self._buffer = deque(maxlen=max(round(delta_second * fps), 1))

    def update(
            self,
            result: Results
    ) -> dict[int, str]:
        if result.boxes.id is None:
            return {}

        if len(self._buffer) == 0:
            self._buffer.appendleft(result)
            return {idx: 'unknown' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            if self._res.get(idx):
                ret[idx] = self._res[idx]
                continue

            if result.boxes.cls[i] not in self._cls_indices:
                continue

            curr_x, curr_y, w, h = result.boxes.xywh[i]

            cal_flag = False
            points = [[curr_x, curr_y]]  # [[x, y], ...]
            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    prev_x, prev_y = buf.boxes.xywh[retrieve[0]][:2]
                    points.append([prev_x, prev_y])

                    delta_track = LineString(points[:2])
                    if j == 0:
                        if delta_track.intersects(self._line):
                            cal_flag = True
                        else:
                            break

            if cal_flag:
                points = np.array(points)
                xs, ys = points[:, 0], points[:, 1]

                try:
                    tan, intercept = np.polyfit(xs, ys, 1)

                    if tan == 0:
                        car_length = w
                    else:
                        sin = math.cos(math.atan(tan)) * tan
                        car_length = abs(h / sin)

                except ValueError:  # vertical vector
                    car_length = h

                if car_length < self._thresholds[0]:
                    size = 'small'
                elif car_length < self._thresholds[1]:
                    size = 'medium'
                else:
                    size = 'large'

                self._res[idx] = size
                ret[idx] = size

        self._buffer.appendleft(result)

        return ret


class ColorClassifier:
    def __init__(
            self,
            cls_indices: list,
            resize: list,
            min_size: list,
            weights='weights/yolo11n-cls-color.pt',
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
            if result.boxes.cls[i] not in self._cls_indices:
                continue

            *_, w, h = result.boxes.xywh[i]
            if w < self._min_size[0] or h < self._min_size[1]:
                if self._res.get(idx):
                    ret[idx] = self._res[idx]
                continue

            x1, y1, x2, y2 = list(map(int, result.boxes.xyxy[i]))
            box_img = img[y1: y2, x1: x2, :]
            box_img = self._transform(Image.fromarray(box_img))

            idxes.append(idx)
            box_imgs.append(box_img)

        preds = self._model.predict(torch.stack(box_imgs), verbose=False, device=0) if box_imgs else {}

        for i, pred in enumerate(preds):
            label = self._idx2cls[pred.probs.top1]
            self._res[idxes[i]] = label
            ret[idxes[i]] = label

        return ret


class VolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(det_zone)

    def update(
            self,
            result: Results
    ) -> int:
        for i, xywh in enumerate(result.boxes.xywh):
            if result.boxes.cls[i] not in self._cls_indices:
                continue

            center = Point(xywh[:2])
            if self._polygon.contains(center):
                self._id_set.add(result.boxes.id[i])

        return len(self._id_set)


class SpeedDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            lengths_m: list,
            angle: float,
            delta_second: float,
            fps: float
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(det_zone)
        self._fps = fps
        self._buffer = deque(maxlen=max(round(delta_second * fps), 1))

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

        self._homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

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
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self._cls_indices or not self._polygon.contains(center):
                continue

            speed = 0

            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_x, prev_y = buf.boxes.xywh[list_idx][:2]
                    curr_x, curr_y = result.boxes.xywh[i][:2]

                    points = np.array([[curr_x, curr_y], [prev_x, prev_y]])
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
        det_zone: list,
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
            upper = LineString([det_zone[0], det_zone[3]])
            lower = LineString([det_zone[1], det_zone[2]])
            line = LineString([(xyxy[0], xyxy[3]), (xyxy[2], xyxy[3])])
            ret[result.boxes.id[i]] = not (line.intersects(lower) or is_between(line, lower, upper))

    return ret


class ParkingDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            displacement_threshold: float,
            fps: float
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(det_zone)
        self._displacement_threshold = displacement_threshold
        self._fps = fps
        self._buffer = deque(maxlen=max(round(delta_second * fps), 1))

    def update(
            self,
            result: Results
    ) -> dict[int, str]:
        if result.boxes.id is None:
            return {}

        if len(self._buffer) == 0:
            self._buffer.append(result)
            return {idx: 'unknown' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            if result.boxes.cls[i] not in self._cls_indices:
                continue

            state = 'unknown'

            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    curr_point = result.boxes.xywh[i][:2]

                    motion_vector = curr_point - prev_point
                    displacement = np.linalg.norm(motion_vector)

                    if displacement < self._displacement_threshold:
                        center = Point(result.boxes.xywh[i][:2])

                        if self._polygon.contains(center):
                            state = 'illegally parked'
                        else:
                            state = 'legally parked'

                    else:
                        state = 'moving'

                    break

            ret[idx] = state

        self._buffer.append(result)

        return ret


class WrongwayDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            fps: float,
            correct_way='up'
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(det_zone)
        self._fps = fps
        self._buffer = deque(maxlen=max(round(delta_second * fps), 1))
        self._correct_way = correct_way

    def update(
            self,
            result: Results
    ) -> dict[int, dict]:
        if result.boxes.id is None:
            return {}

        if len(self._buffer) == 0:
            self._buffer.append(result)
            return {idx: {'vector': [0., 0.], 'wrongway': False} for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self._cls_indices or not self._polygon.contains(center):
                continue

            motion_vector = [0., 0.]

            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    curr_point = result.boxes.xywh[i][:2]

                    motion_vector = curr_point - prev_point
                    norm = np.linalg.norm(motion_vector)
                    motion_vector /= norm
                    motion_vector = list(motion_vector)
                    break

            if self._correct_way == 'up':
                wrongway = motion_vector[1] > 0
            elif self._correct_way == 'down':
                wrongway = motion_vector[1] < 0
            elif self._correct_way == 'left':
                wrongway = motion_vector[0] < 0
            elif self._correct_way == 'right':
                wrongway = motion_vector[0] > 0
            else:
                wrongway = False

            ret[idx] = {'vector': motion_vector, 'wrongway': wrongway}

        self._buffer.append(result)

        return ret


class LanechangeDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            solid_lines: list,
            fps: float
    ):
        self._id_set = set()
        self._cls_indices = cls_indices
        self._polygon = Polygon(det_zone)
        self._fps = fps
        self._buffer = deque(maxlen=max(round(delta_second * fps), 1))
        self._solid_lines = [LineString(solid_line) for solid_line in solid_lines]

    def update(
            self,
            result: Results
    ) -> dict[int, bool]:
        if result.boxes.id is None:
            return {}

        if len(self._buffer) == 0:
            self._buffer.append(result)
            return {idx: False for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self._cls_indices or not self._polygon.contains(center):
                continue

            lanechange = False

            for j, buf in enumerate(self._buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    next_point = 2 * result.boxes.xywh[i][:2] - prev_point

                    track = LineString([prev_point, next_point])
                    for solid_line in self._solid_lines:
                        if solid_line.intersects(track):
                            lanechange = True
                            break

                    break

            ret[idx] = lanechange

        self._buffer.append(result)

        return ret
