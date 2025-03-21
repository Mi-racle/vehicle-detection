import math
from collections import deque
from copy import deepcopy
from time import time
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

from utils import cal_euclidean_distance, cal_intersection_points, cal_homography_matrix, reproject, \
    cal_intersection_ratio, update_counts, generate_video_generally, increment_path, \
    generate_video, generate_hash20


class JamDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            n_threshold: float,
            video_length: float,
            fps: float
    ):
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__n_threshold = n_threshold
        self.__fps = fps
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(maxlen=self.__video_frame_num)
        self.__jam = False
        self.__countdown = self.__video_frame_num

    def update(
            self,
            result: Results
    ) -> bool:
        # TODO
        self.__result_buffer.append(result)

        if self.__jam:
            self.__countdown -= 1
            return True

        count = 0

        if result.boxes.id is None:
            return False

        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not Polygon(self.__det_zone).contains(center):
                continue

            count += 1

            if count >= self.__n_threshold:
                self.__jam = True
                return True

        return False

    def plot(
            self,
            state: bool,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(
            img,
            f'Jam: {state}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        if self.__countdown <= 0:
            dest = generate_hash20(f'{type(self).__name__}{time()}')
            dests.append(dest)

            generate_video_generally(
                f'{output_dir}/{dest}.mp4',
                self.__result_buffer,
                self.__fps,
                color=(0, 0, 255)
            )

            self.__countdown = self.__video_frame_num
            self.__jam = False

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class QueueDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            lengths_m: list,
            angle: float,
            sampling_period: float,
            fps: float
    ):
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__sampling_frame_num = max(round(sampling_period * fps), 1)
        self.__fps = fps
        self.__buffered_result: Results | None = None
        self.__countdown = self.__sampling_frame_num
        self.__queue_length = 0

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

        self.__homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results
    ) -> int:
        self.__buffered_result = result

        head_point = ()
        tail_point = ()

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            bm = (result.boxes.xywh[i][0], xyxy[-1])

            # TODO
            if Polygon(self.__det_zone).contains(Point(bm)):
                head_point = bm if not head_point else (head_point if head_point[1] <= bm[1] else bm)
                tail_point = bm if not tail_point else (tail_point if tail_point[1] > bm[1] else bm)

        if head_point and tail_point:
            points = np.array([head_point, tail_point])
            repro_points = reproject(points, self.__homography_matrix)
            self.__queue_length = cal_euclidean_distance(repro_points[0], repro_points[1])

        else:
            self.__queue_length = 0

        self.__countdown -= 1

        return self.__queue_length

    def plot(
            self,
            state: int,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__buffered_result)

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=5)
        cv2.putText(
            img,
            f'Queue: {state}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash20(f'{type(self).__name__}{time()}')
            dests.append(dest)

            cv2.polylines(frame_copy, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame_copy,
                f'Queue: {self.__queue_length}',
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0)
            )

            self.__countdown = self.__sampling_frame_num

            cv2.imwrite(increment_path(f'{output_dir}/{dest}.jpg'), frame_copy)

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class DensityDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            length_m: float,
            sampling_period: float,
            fps: float
    ):
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__length_m = length_m
        self.__sampling_frame_num = max(round(sampling_period * fps), 1)
        self.__fps = fps
        self.__buffered_result: Results | None = None
        self.__countdown = self.__sampling_frame_num
        self.__density = 0

    def update(
            self,
            result: Results
    ) -> int:
        self.__buffered_result = result
        self.__countdown -= 1
        self.__density = 0

        if self.__length_m <= 0:
            return self.__density

        volume = 0

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            bm = Point((result.boxes.xywh[i][0], xyxy[-1]))

            if Polygon(self.__det_zone).contains(bm):
                volume += 1

        self.__density = volume / self.__length_m * 1e3

        return self.__density

    def plot(
            self,
            state: int,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__buffered_result)

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=5)
        cv2.putText(
            img,
            f'Density: {state}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash20(f'{type(self).__name__}{time()}')
            dests.append(dest)

            cv2.polylines(frame_copy, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame_copy,
                f'Density: {self.__density}',
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0)
            )

            self.__countdown = self.__sampling_frame_num

            cv2.imwrite(increment_path(f'{output_dir}/{dest}.jpg'), frame_copy)

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class SizeDetector:
    def __init__(
            self,
            cls_indices: list,
            vertices: list,
            thresholds: list,
            delta_second: float,
            fps: float
    ):
        self.__res = {}
        self.__cls_indices = cls_indices
        self.__line = LineString(vertices)
        self.__thresholds = thresholds
        self.__buffer = deque(maxlen=max(round(delta_second * fps), 1))

    def update(
            self,
            result: Results
    ) -> dict[int, str]:
        if result.boxes.id is None:
            return {}

        if len(self.__buffer) == 0:
            self.__buffer.appendleft(result)
            return {idx: 'unknown' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            if self.__res.get(idx):
                ret[idx] = self.__res[idx]
                continue

            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            w, h = result.boxes.xywh[i][-2:]
            curr_x, curr_y, br_x, br_y = result.boxes.xyxy[i]

            cal_flag = False
            for j, buf in enumerate(self.__buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    prev_x, prev_y = buf.boxes.xyxy[retrieve[0]][:2]
                    delta_track = LineString([[prev_x, prev_y], [curr_x, curr_y]])

                    if delta_track.intersects(self.__line):
                        cal_flag = True
                        break

            if cal_flag:
                car_length = h

                if car_length < self.__thresholds[0]:
                    size = 'small'
                elif car_length < self.__thresholds[1]:
                    size = 'medium'
                else:
                    size = 'large'

                self.__res[idx] = size
                ret[idx] = size

        self.__buffer.appendleft(result)

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
        self.__res = {}
        self.__cls_indices = cls_indices
        self.__idx2cls: dict = yaml.safe_load(open(cls_path, 'r'))
        self.__model = YOLO(weights)
        self.__transform = transforms.Compose([
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.__min_size = min_size

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
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            *_, w, h = result.boxes.xywh[i]
            if w < self.__min_size[0] or h < self.__min_size[1]:
                if self.__res.get(idx):
                    ret[idx] = self.__res[idx]
                continue

            x1, y1, x2, y2 = list(map(int, result.boxes.xyxy[i]))
            box_img = img[y1: y2, x1: x2, :]
            box_img = self.__transform(Image.fromarray(box_img))

            idxes.append(idx)
            box_imgs.append(box_img)

        preds = self.__model.predict(torch.stack(box_imgs), verbose=False, device=0) if box_imgs else {}

        for i, pred in enumerate(preds):
            label = self.__idx2cls[pred.probs.top1]
            self.__res[idxes[i]] = label
            ret[idxes[i]] = label

        return ret


class SectionDetector:
    """ Section Volume Detector """
    def __init__(
            self,
            cls_indices: list,
            det_line: list,
            video_length: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_line = det_line
        self.__fps = fps
        self.__delta_frame_num = 3  # tentative
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__text_buffer = deque(maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__countdown = self.__video_frame_num

    def update(
            self,
            result: Results,
    ) -> int:
        self.__result_buffer.append(result)

        if len(self.__result_buffer) == 0:
            self.__text_buffer.append(f'Section volume: {len(self.__id_set)}')
            return len(self.__id_set)

        ids = result.boxes.id if result.boxes.id is not None else []
        for i, idx in enumerate(ids):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            if idx in self.__id_set:
                continue

            for j, buf in enumerate(list(self.__result_buffer)[-self.__delta_frame_num:]):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_center = buf.boxes.xywh[list_idx][:2]
                    curr_center = result.boxes.xywh[i][:2]

                    delta_track = LineString([prev_center, curr_center])
                    if delta_track.intersects(LineString(self.__det_line)):
                        self.__id_set.add(idx)
                        break

        self.__text_buffer.append(f'Section volume: {len(self.__id_set)}')
        self.__countdown -= 1

        return len(self.__id_set)

    def plot(
            self,
            state: int,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.line(img, self.__det_line[0], self.__det_line[1], color=(0, 255, 0), thickness=2)
        cv2.putText(
            img,
            f'Section volume: {state}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        if self.__countdown <= 0:
            dest = generate_hash20(f'{type(self).__name__}{time()}')
            dests.append(dest)

            generate_video_generally(
                f'{output_dir}/{dest}.mp4',
                self.__result_buffer,
                self.__fps,
                self.__det_line,
                self.__text_buffer,
                (0, 255, 0)
            )

            self.__countdown = self.__video_frame_num
            self.__id_set.clear()

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class VelocityDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            lengths_m: list,
            angle: float,
            delta_second: float,
            video_length: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__velocity_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__velocity_counts = {}
        self.__output_countdowns = {}

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

        self.__homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results
    ) -> dict[float, str]:
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            self.__velocity_buffer.append({})
            return {}

        if len(self.__result_buffer) == 0:
            self.__velocity_buffer.append({idx: f'0.0 km/h' for idx in result.boxes.id})
            return {idx: '0.0 km/h' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not Polygon(self.__det_zone).contains(center):
                continue

            velocity = 0

            for j, buf in enumerate(self.__result_buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_x, prev_y = buf.boxes.xywh[list_idx][:2]
                    curr_x, curr_y = result.boxes.xywh[i][:2]

                    points = np.array([[curr_x, curr_y], [prev_x, prev_y]])
                    repro_points = reproject(points, self.__homography_matrix)
                    distance = cal_euclidean_distance(repro_points[0], repro_points[1])
                    velocity = distance / ((len(self.__result_buffer) - j) / self.__fps) * 3.6

                    break

            ret[idx] = f'{velocity:.3f} km/h'

            if idx not in self.__id_set:
                update_counts(
                    True,
                    True,
                    idx,
                    self.__velocity_counts,
                    self.__output_countdowns,
                    0,
                    self.__video_frame_num
                )

        self.__velocity_buffer.append(ret)

        return ret

    def plot(
            self,
            states: dict[float, str],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 255, 0)
            )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num,
                    self.__velocity_buffer
                )

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class VolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            sampling_period: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__sampling_frame_num = max(round(sampling_period * fps), 1)
        self.__fps = fps
        self.__buffered_result: Results | None = None
        self.__countdown = self.__sampling_frame_num
        self.__volume = 0

    def update(
            self,
            result: Results
    ) -> int:
        self.__buffered_result = result
        self.__volume = 0

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            center = Point(result.boxes.xywh[i][:2])

            if Polygon(self.__det_zone).contains(center):
                self.__volume += 1

        self.__countdown -= 1

        return self.__volume

    def plot(
            self,
            state: int,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__buffered_result)

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=5)
        cv2.putText(
            img,
            f'Intersection volume: {state}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash20(f'{type(self).__name__}{time()}')
            dests.append(dest)

            cv2.polylines(frame_copy, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame_copy,
                f'Intersection volume: {self.__volume}',
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0)
            )

            self.__countdown = self.__sampling_frame_num

            cv2.imwrite(increment_path(f'{output_dir}/{dest}.jpg'), frame_copy)

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class PimDetector:
    def __init__(
            self,
            cls_indices: list,
            motor_zone: list,
            iou_threshold: float,
            duration_threshold: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__motor_zone = motor_zone
        self.__iou_threshold = iou_threshold
        self.__fps = fps
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__result_buffer = deque(maxlen=self.__duration_frame_num)
        self.__pim_counts = {}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return {}

        ret = {}

        vehicle_retrieves = np.where(result.boxes.cls == self.__cls_indices[1])[0]

        for i, idx in enumerate(result.boxes.id):
            if result.boxes.cls[i] not in self.__cls_indices[0:]:
                continue

            xyxy = result.boxes.xyxy[i]
            xywh = result.boxes.xywh[i]

            walking = True
            piming = False

            for retrieve in vehicle_retrieves:
                iou = cal_intersection_ratio(xyxy, result.boxes.xyxy[retrieve])

                if iou > self.__iou_threshold:
                    walking = False
                    break

            if walking:
                piming = Polygon(self.__motor_zone).contains(Point((xywh[0], xyxy[3])))  # bottom mid

            ret[idx] = piming

            if idx not in self.__id_set:
                update_counts(
                    piming,
                    True,
                    idx,
                    self.__pim_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    1  # image instead of video
                )

        return ret

    def plot(
            self,
            states: dict[float, str],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__motor_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            cv2.putText(
                img,
                'breaking in' if state else '',
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                color=(0, 0, 255) if state else (0, 255, 0)
            )
            cv2.rectangle(
                img,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color=(0, 0, 255) if state else (0, 255, 0),
                thickness=2
            )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []
        result = self.__result_buffer[-1]

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                retrieve = np.where(result.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    frame_copy = result.orig_img.copy()
                    list_idx = retrieve[0]

                    xyxy = result.boxes.xyxy[list_idx]

                    cv2.rectangle(
                        frame_copy,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255),
                        thickness=2
                    )

                    cv2.imwrite(increment_path(f'{output_dir}/{dest}.jpg'), frame_copy)

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class ParkingDetector:
    def __init__(
            self,
            cls_indices: list,
            noparking_zone: list,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            displacement_threshold: float,
            fps: float
    ):
        self.__model = YOLO('weights/yolo11s.pt')
        self.__cls_indices = cls_indices
        self.__noparking_zone = noparking_zone
        self.__displacement_threshold = displacement_threshold
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num)
        )
        self.__id_set = set()
        self.__parking_counts = {}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results
    ) -> dict[float, str]:
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: 'unknown' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            state = 'unknown'

            for j, buf in enumerate(self.__result_buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    curr_point = result.boxes.xywh[i][:2]

                    motion_vector = curr_point - prev_point
                    displacement = np.linalg.norm(motion_vector)

                    if displacement < self.__displacement_threshold:
                        center = Point(result.boxes.xywh[i][:2])

                        if Polygon(self.__noparking_zone).contains(center):
                            state = 'illegally parked'
                        else:
                            state = 'legally parked'

                    else:
                        state = 'moving'

                    break

            ret[idx] = state

            if idx not in self.__id_set:
                update_counts(
                    state,
                    'illegally parked',
                    idx,
                    self.__parking_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    self.__video_frame_num
                )

        return ret

    def plot(
            self,
            states: dict[float, str],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__noparking_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            color = (0, 0, 255) if state == 'illegally parked' else (0, 255, 0)
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                color
            )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class WrongwayDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            fps: float,
            valid_direction='up'
    ):
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__fps = fps
        self.__valid_direction = valid_direction
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__id_set = set()
        self.__wrongway_counts = {}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results
    ) -> dict[float, dict]:
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: {'vector': [0., 0.], 'wrongway': False} for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            if (
                    result.boxes.cls[i] not in self.__cls_indices or
                    not Polygon(self.__det_zone).contains(Point(result.boxes.xywh[i][:2]))
            ):
                continue

            motion_vector = [0., 0.]

            for j, buf in enumerate(self.__result_buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    curr_point = result.boxes.xywh[i][:2]

                    motion_vector = curr_point - prev_point
                    norm = np.linalg.norm(motion_vector)
                    if norm > 1e-6:
                        motion_vector /= norm
                    motion_vector = list(motion_vector)
                    break

            wrongway = self.__is_wrongway(motion_vector)
            ret[idx] = {'vector': motion_vector, 'wrongway': wrongway}

            if idx not in self.__id_set:
                update_counts(
                    wrongway,
                    True,
                    idx,
                    self.__wrongway_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    self.__video_frame_num
                )

        return ret

    def plot(
            self,
            states: dict[float, dict],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            lstats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            wrongway = state['wrongway']
            cv2.putText(
                img,
                'wrong way' if wrongway else 'right way',
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 0, 255) if wrongway else (0, 255, 0)
            )
            # xywh = result.boxes.xywh[retrieve]
            # vector = state['vector']
            # cv2.arrowedLine(
            #     img,
            #     (int(xywh[0]), int(xywh[1])),
            #     (int(xywh[0] + 50 * vector[0]), int(xywh[1] + 50 * vector[1])),
            #     color=color
            # )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return dests

    def __is_wrongway(self, motion_vector: list[float]) -> bool:
        if self.__valid_direction == 'up':
            return motion_vector[1] > 0
        elif self.__valid_direction == 'down':
            return motion_vector[1] < 0
        elif self.__valid_direction == 'left':
            return motion_vector[0] < 0
        elif self.__valid_direction == 'right':
            return motion_vector[0] > 0
        else:
            return False

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class LanechangeDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            solid_lines: list,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__polygon = Polygon(det_zone)
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__solid_lines = [LineString(solid_line) for solid_line in solid_lines]
        self.__lanechange_counts = {}  # {id: count}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results
    ) -> dict[int, bool]:
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: False for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not Polygon(self.__det_zone).contains(center):
                continue

            lanechange = False

            for j, buf in enumerate(list(self.__result_buffer)[-self.__delta_frame_num:]):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_point = buf.boxes.xywh[list_idx][:2]
                    next_point = 2 * result.boxes.xywh[i][:2] - prev_point

                    track = LineString([prev_point, next_point])
                    for solid_line in self.__solid_lines:
                        if solid_line.intersects(track):
                            lanechange = True
                            break

                    break

            ret[idx] = lanechange

            if idx not in self.__id_set:
                update_counts(
                    lanechange,
                    True,
                    idx,
                    self.__lanechange_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    self.__video_frame_num
                )

        return ret

    def plot(
            self,
            states: dict[float, dict],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for solid_line in self.__solid_lines:
            cv2.line(
                img,
                tuple(map(int, solid_line.coords[0])),
                tuple(map(int, solid_line.coords[-1])),
                color=(255, 255, 255),
                thickness=2
            )

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            cv2.putText(
                img,
                'changing' if state else 'staying',
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 0, 255) if state else (0, 255, 0)
            )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class SpeedingDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            lengths_m: list,
            angle: float,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            speed_threshold: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__speed_threshold = speed_threshold
        self.__velocity_counts = {}
        self.__output_countdowns = {}

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

        self.__homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results
    ) -> dict[float, str]:
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: 'not speeding (0 km/h)' for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not Polygon(self.__det_zone).contains(center):
                continue

            velocity = 0
            speeding = False

            for j, buf in enumerate(self.__result_buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]
                    prev_x, prev_y = buf.boxes.xywh[list_idx][:2]
                    curr_x, curr_y = result.boxes.xywh[i][:2]

                    points = np.array([[curr_x, curr_y], [prev_x, prev_y]])
                    repro_points = reproject(points, self.__homography_matrix)
                    distance = cal_euclidean_distance(repro_points[0], repro_points[1])
                    velocity = distance / ((len(self.__result_buffer) - j) / self.__fps) * 3.6
                    speeding = velocity >= self.__speed_threshold

                    break

            ret[idx] = ('speeding' if speeding else 'not speeding') + f' ({velocity:.3f} km/h)'

            if idx not in self.__id_set:
                update_counts(
                    speeding,
                    True,
                    idx,
                    self.__velocity_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    self.__video_frame_num
                )

        return ret

    def plot(
            self,
            states: dict[float, str],
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in states:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = states[idx]
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 255, 0) if 'not speeding' in state else (0, 0, 255)
            )

        return img

    def output_corpus(self, output_dir: str) -> list[str]:
        dests = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash20(f'{type(self).__name__}{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return dests

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1
progress