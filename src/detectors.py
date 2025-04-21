import math
import re
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from time import time
from typing import Any

import cv2
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from ultralytics.engine.results import Results

from utils import cal_euclidean_distance, cal_intersection_points, cal_homography_matrix, reproject, \
    cal_intersection_ratio, update_counts, generate_video_generally, increment_path, \
    generate_video, generate_hash, put_text_ch


class Detector(ABC):
    @abstractmethod
    def update(self, result: Results | tuple[list, list] | tuple[list, list[list[str | float]]]): ...

    @abstractmethod
    def plot(self, frame, stats_line: int | None = 1, subscript_line: int | None = 1): ...

    @abstractmethod
    def output_corpus(self, output_dir: str, orig_img=None): ...

    @staticmethod
    def update_line(stats: int, subscript: int): ...


class JamDetector(Detector):
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
    ):
        self.__result_buffer.append(result)

        if self.__jam:
            self.__countdown -= 1
            return

        count = 0

        if result.boxes.id is None:
            return

        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not Polygon(self.__det_zone).contains(center):
                continue

            count += 1

            if count >= self.__n_threshold:
                self.__jam = True
                return

    def plot(
            self,
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
            f'Jam: {self.__jam}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        if self.__countdown <= 0:
            dest = generate_hash(f'{type(self).__name__}{time()}') + '.mp4'
            corpus_infos.append({'dest': dest, 'plate_no': None})

            generate_video_generally(
                f'{output_dir}/{dest}',
                self.__result_buffer,
                self.__fps,
                color=(0, 0, 255)
            )

            self.__countdown = self.__video_frame_num
            self.__jam = False

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class QueueDetector(Detector):
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
        self.__queue_length = 0
        self.__countdown = self.__sampling_frame_num

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
    ):
        self.__buffered_result = result

        head_point = ()
        tail_point = ()

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            bm = (result.boxes.xywh[i][0], xyxy[-1])

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

    def plot(
            self,
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
            f'Queue: {self.__queue_length}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash(f'{type(self).__name__}{time()}') + '.jpg'
            corpus_infos.append({'dest': dest, 'plate_no': None})

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

            cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class DensityDetector(Detector):
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
    ):
        self.__buffered_result = result
        self.__countdown -= 1
        self.__density = 0

        if self.__length_m <= 0:
            return

        volume = 0

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            bm = Point((result.boxes.xywh[i][0], xyxy[-1]))

            if Polygon(self.__det_zone).contains(bm):
                volume += 1

        self.__density = volume / self.__length_m * 1e3

    def plot(
            self,
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
            f'Density: {self.__density}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash(f'{type(self).__name__}{time()}') + '.jpg'
            corpus_infos.append({'dest': dest, 'plate_no': None})

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

            cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class SizeDetector(Detector):
    def __init__(
            self,
            cls_indices: list,
            det_line: list,
            thresholds: list,
            delta_second: float,
            fps: float
    ):
        self.__history_size = {}  # TODO recycle
        self.__cls_indices = cls_indices
        self.__det_line = det_line
        self.__thresholds = thresholds
        self.__result_buffer = deque(maxlen=max(round(delta_second * fps), 1))
        self.__counts = {}
        self.__output_countdowns = {}
        self.__ret: dict[float, str] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

        for i, idx in enumerate(result.boxes.id):
            if idx in self.__history_size:
                self.__ret[idx] = self.__history_size[idx]
                continue

            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            w, h = result.boxes.xywh[i][-2:]
            curr_x, curr_y, br_x, br_y = result.boxes.xyxy[i]

            cal_flag = False
            for j, buf in enumerate(self.__result_buffer):
                if buf.boxes.id is None:
                    continue

                retrieve = np.where(buf.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    prev_x, prev_y = buf.boxes.xyxy[retrieve[0]][:2]
                    delta_track = LineString([[prev_x, prev_y], [curr_x, curr_y]])

                    if delta_track.intersects(LineString(self.__det_line)):
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

                if idx not in self.__history_size:
                    update_counts(
                        True,
                        True,
                        idx,
                        self.__counts,
                        self.__output_countdowns,
                        1,
                        1  # image instead of video
                    )

                self.__history_size[idx] = size
                self.__ret[idx] = size

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.line(img, self.__det_line[0], self.__det_line[1], color=(255, 0, 0), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
            color = (200, 0, 0) if state == 'small' else (
                (0, 200, 0) if state == 'medium' else (0, 0, 200)
            )
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                color=color
            )
            cv2.rectangle(
                img,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color=color,
                thickness=2
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        result = self.__result_buffer[-1]

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.jpg'
                corpus_infos.append({'dest': dest, 'plate_no': None})

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

                    cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class SectionDetector(Detector):
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
    ):
        self.__result_buffer.append(result)

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

    def plot(
            self,
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
            f'Section volume: {len(self.__id_set)}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        if self.__countdown <= 0:
            dest = generate_hash(f'{type(self).__name__}{time()}') + '.mp4'
            corpus_infos.append({'dest': dest, 'plate_no': None})

            generate_video_generally(
                f'{output_dir}/{dest}',
                self.__result_buffer,
                self.__fps,
                self.__det_line,
                self.__text_buffer,
                (0, 255, 0)
            )

            self.__countdown = self.__video_frame_num
            self.__id_set.clear()

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class VelocityDetector(Detector):
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
        self.__ret: dict[float, str] = {}

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
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            self.__velocity_buffer.append({})
            return

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

            self.__ret[idx] = f'{velocity:.3f} km/h'

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

        self.__velocity_buffer.append(self.__ret.copy())

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 255, 0)
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.mp4'
                corpus_infos.append({'dest': dest, 'plate_no': None})

                generate_video(
                    f'{output_dir}/{dest}',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num,
                    self.__velocity_buffer
                )

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class VolumeDetector(Detector):
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
    ):
        self.__buffered_result = result
        self.__countdown -= 1
        self.__volume = 0

        if result.boxes.id is None:
            return

        for i, idx in enumerate(result.boxes.id):
            if result.boxes.cls[i] not in self.__cls_indices:
                continue

            center = Point(result.boxes.xywh[i][:2])

            if Polygon(self.__det_zone).contains(center):
                self.__volume += 1

    def plot(
            self,
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
            f'Intersection volume: {self.__volume}',
            (0, 30 * stats_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0)
        )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        frame_copy = self.__buffered_result.orig_img.copy()

        if self.__countdown <= 0:
            dest = generate_hash(f'{type(self).__name__}{time()}') + '.jpg'
            corpus_infos.append({'dest': dest, 'plate_no': None})

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

            cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats + 1, subscript


class PimDetector(Detector):
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
        self.__ret: dict[float, bool] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

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

            self.__ret[idx] = piming

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

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__motor_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
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

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        result = self.__result_buffer[-1]

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.jpg'
                corpus_infos.append({'dest': dest, 'plate_no': None})

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

                    cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class ParkingDetector(Detector):
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
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__noparking_zone = noparking_zone
        self.__displacement_threshold = displacement_threshold
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__parking_counts = {}
        self.__output_countdowns = {}
        self.__ret: dict[float, str] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

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

            self.__ret[idx] = state

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

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__noparking_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
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

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.mp4'
                corpus_infos.append({'dest': dest, 'plate_no': None})

                generate_video(
                    f'{output_dir}/{dest}',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class WrongwayDetector(Detector):
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
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__det_zone = det_zone
        self.__fps = fps
        self.__valid_direction = valid_direction
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__wrongway_counts = {}
        self.__output_countdowns = {}
        self.__ret: dict[float, dict] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

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
            self.__ret[idx] = {'vector': motion_vector, 'wrongway': wrongway}

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

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            lstats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
            wrongway = state['wrongway']
            cv2.putText(
                img,
                'wrong way' if wrongway else 'right way',
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 0, 255) if wrongway else (0, 255, 0)
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.mp4'
                corpus_infos.append({'dest': dest, 'plate_no': None})

                generate_video(
                    f'{output_dir}/{dest}',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return corpus_infos

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


class LanechangeDetector(Detector):
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
        self.__ret: dict[float, bool] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

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

            self.__ret[idx] = lanechange

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

    def plot(
            self,
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

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
            cv2.putText(
                img,
                'changing' if state else 'staying',
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 0, 255) if state else (0, 255, 0)
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.mp4'
                corpus_infos.append({'dest': dest, 'plate_no': None})

                generate_video(
                    f'{output_dir}/{dest}',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class SpeedingDetector(Detector):
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
        self.__ret: dict[float, str] = {}

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
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

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

            self.__ret[idx] = ('speeding' if speeding else 'not speeding') + f' ({velocity:.3f} km/h)'

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

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        cv2.polylines(img, np.array([self.__det_zone]), isClosed=True, color=(0, 0, 255), thickness=2)

        for idx in self.__ret:
            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            state = self.__ret[idx]
            cv2.putText(
                img,
                state,
                (int(xyxy[2]), int(xyxy[1] + 12 * subscript_line)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .4,
                (0, 255, 0) if 'not speeding' in state else (0, 0, 255)
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.mp4'
                corpus_infos.append({'dest': dest, 'plate_no': None})

                generate_video(
                    f'{output_dir}/{dest}',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class PlateDetector(Detector):
    def __init__(
            self,
            duration_threshold: float,
            fps: float
    ):
        self.__plate_set = set()
        self.__fps = fps
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__result_buffer = deque(maxlen=self.__duration_frame_num)
        self.__plate_counts = {}
        self.__output_countdowns = {}
        self.__ret: dict[int, str] = {}

    def update(
            self,
            result: tuple[list, list] | tuple[list, list[list[str | float]]]
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        bboxes, rec_reses = result
        txts = [rec_reses[i][0] for i in range(len(rec_reses))]
        # scores = [rec_reses[i][1] for i in range(len(rec_reses))]

        for i, bbox in enumerate(bboxes):
            txt: str = txts[i]
            txt = self.__regularize(txt)

            if not txt:  # txt is not valid plate license
                continue

            self.__ret[i] = txt

            if txt not in self.__plate_set:
                update_counts(
                    True,
                    True,
                    txt,
                    self.__plate_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    1  # image instead of video
                )

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = self.__result_buffer[-1]

        for idx in self.__ret:
            bbox = result[0][idx]
            tl, br = list(map(int, bbox[0])), list(map(int, bbox[2]))
            state = self.__ret[idx]
            frame = put_text_ch(
                frame,
                state,
                (int(br[0]), int(tl[1] + 12 * subscript_line)),
                .8,
                color=(0, 0, 255)
            )
            cv2.rectangle(frame, tl, br, color=(0, 0, 255), thickness=2)

        return frame

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []  # [{'dest': ..., 'plate_no': ...}]
        bboxes = self.__result_buffer[-1][0]

        for plate in list(self.__output_countdowns.keys()):
            self.__output_countdowns[plate] -= 1

            if self.__output_countdowns[plate] <= 0:
                del self.__output_countdowns[plate]
                self.__plate_set.add(plate)

                dest = generate_hash(f'{type(self).__name__}{plate}{time()}') + '.jpg'
                corpus_infos.append({'dest': dest, 'plate_no': plate})

                key = next((k for k, v in self.__ret.items() if v == plate), None)

                if key and orig_img is not None:
                    frame_copy = orig_img.copy()

                    bbox = bboxes[key]
                    tl, br = list(map(int, bbox[0])), list(map(int, bbox[2]))

                    frame_copy = put_text_ch(
                        frame_copy,
                        plate,
                        (int(br[0]), int(tl[1] + 12)),
                        .8,
                        color=(0, 0, 255)
                    )
                    cv2.rectangle(frame_copy, tl, br, (0, 0, 255), thickness=2)

                    cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def __regularize(txt: str):
        txt = txt.replace('·', '')
        txt = txt.replace('l', '1')
        txt = txt.replace('I', '1')
        txt = txt.replace('O', '0')
        txt = txt.replace('o', '0')

        if not re.match(
                r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]$',
                txt
        ):
            return ''

        return txt

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript + 1


class TriangleDetector(Detector):
    def __init__(
            self,
            duration_threshold: float,
            fps: float
    ):
        self.__id_set = set()
        self.__fps = fps
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__result_buffer = deque(maxlen=self.__duration_frame_num)
        self.__triangle_counts = {}
        self.__output_countdowns = {}
        self.__ret: dict[float, bool] = {}

    def update(
            self,
            result: Results
    ):
        self.__result_buffer.append(result)
        self.__ret = {}

        if result.boxes.id is None:
            return

        for i, idx in enumerate(result.boxes.id):
            triangle = True
            self.__ret[idx] = triangle

            if idx not in self.__id_set:
                update_counts(
                    triangle,
                    True,
                    idx,
                    self.__triangle_counts,
                    self.__output_countdowns,
                    self.__duration_frame_num,
                    1  # image instead of video
                )

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        for idx in self.__ret:
            if not self.__ret[idx]:
                continue

            retrieve = np.where(result.boxes.id == idx)[0][0]
            xyxy = result.boxes.xyxy[retrieve]
            cv2.rectangle(
                img,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color=(0, 0, 255),
                thickness=2
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        result = self.__result_buffer[-1]

        for idx in list(self.__output_countdowns.keys()):
            self.__output_countdowns[idx] -= 1

            if self.__output_countdowns[idx] <= 0:
                del self.__output_countdowns[idx]
                self.__id_set.add(idx)

                dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.jpg'
                corpus_infos.append({'dest': dest, 'plate_no': None})

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

                    cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_copy)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript


class ObjectDetector(Detector):
    def __init__(
            self,
            cls_indices: list,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__fps = fps
        self.__result_buffer = []
        self.__ret: dict[float, np.ndarray] = {}

    def update(
            self,
            result: Results
    ):
        self.__ret = {}
        self.__result_buffer.append(result)

        if result.boxes.id is None:
            return

        for i, idx in enumerate(result.boxes.id):
            if result.boxes.cls[i] not in self.__cls_indices or idx in self.__id_set:
                continue

            frame_height, frame_width = result.orig_img.shape[-3: -1]
            xyxy = result.boxes.xyxy[i]

            if (
                xyxy[0] > 10 and
                xyxy[1] > 10 and
                xyxy[2] < frame_width - 10 and
                xyxy[3] < frame_height - 10
            ):
                self.__ret[idx] = xyxy

    def plot(
            self,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None = None,
            stats_line: int | None = 1,
            subscript_line: int | None = 1
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])

        if frame is not None:
            result.orig_img = frame

        img = result.plot(conf=False, line_width=1)

        for idx in self.__ret:
            xyxy = self.__ret[idx]
            cv2.rectangle(
                img,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color=(0, 0, 255),
                thickness=1
            )

        return img

    def output_corpus(self, output_dir: str, orig_img=None) -> list[dict[str, str | None]]:
        corpus_infos = []
        result = self.__result_buffer[-1]

        for idx in self.__ret:
            self.__id_set.add(idx)

            dest = generate_hash(f'{type(self).__name__}{idx}{time()}') + '.jpg'
            corpus_infos.append({'dest': dest, 'plate_no': None})

            xyxy = self.__ret[idx]

            frame_out = result.orig_img[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]

            cv2.imwrite(increment_path(f'{output_dir}/{dest}'), frame_out)

        return corpus_infos

    @staticmethod
    def update_line(stats: int, subscript: int):
        return stats, subscript
