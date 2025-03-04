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
    cal_intersection_ratio, generate_videos_respectively, update_counts, generate_video_generally, increment_path, \
    generate_video, generate_hash20


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


class VolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            det_line: list,
            video_length: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__vertices = det_line
        self.__det_line = LineString(det_line)
        self.__fps = fps
        self.__delta_frame_num = 3  # tentative
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__frame_buffer = deque(maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__text_buffer = deque(maxlen=max(self.__delta_frame_num, self.__video_frame_num))
        self.__countdown = self.__video_frame_num

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
    ) -> int:
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

        if len(self.__result_buffer) == 0:
            self.__text_buffer.append(f'Volume: {len(self.__id_set)}')
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
                    if delta_track.intersects(self.__det_line):
                        self.__id_set.add(idx)
                        break

        self.__text_buffer.append(f'Volume: {len(self.__id_set)}')
        self.__countdown -= 1

        if self.__countdown <= 0:
            print(self.__text_buffer)
            generate_video_generally(
                f'runs/volume',
                self.__frame_buffer,
                self.__fps,
                self.__vertices,
                self.__text_buffer,
                (0, 255, 0)
            )

            self.__countdown = self.__video_frame_num
            self.__id_set.clear()

        return len(self.__id_set)


class VelocityDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            lengths_m: list,
            angle: float,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__polygon = Polygon(det_zone)
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__frame_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__velocity_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
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

        self._homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
    ) -> dict[int, float]:
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

        if result.boxes.id is None:
            self.__velocity_buffer.append({})
            return {}

        if len(self.__result_buffer) == 0:
            self.__velocity_buffer.append({idx: f'0.0 km/h' for idx in result.boxes.id})
            return {idx: 0 for idx in result.boxes.id}

        ret = {}
        texts = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not self.__polygon.contains(center):
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
                    repro_points = reproject(points, self._homography_matrix)
                    distance = cal_euclidean_distance(repro_points[0], repro_points[1])
                    velocity = distance / ((len(self.__result_buffer) - j) / self.__fps) * 3.6
                    break

            ret[idx] = velocity
            texts[idx] = f'{velocity:.3f} km/h'

            update_counts(
                True,
                True,
                idx,
                self.__velocity_counts,
                self.__output_countdowns,
                self.__duration_frame_num,
                self.__video_frame_num
            )

        generate_videos_respectively(
            f'runs/velocity',
            self.__result_buffer,
            self.__frame_buffer,
            self.__fps,
            self.__video_frame_num,
            self.__output_countdowns,
            self.__velocity_buffer
        )

        self.__velocity_buffer.append(texts)

        return ret


class PolumeDetector:
    def __init__(
            self,
            cls_indices: list,
            iou_threshold: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__iou_threshold = iou_threshold

    def update(
            self,
            result: Results
    ) -> int:
        vehicle_retrieves = np.where(result.boxes.cls == self.__cls_indices[1])[0]

        for i, xyxy in enumerate(result.boxes.xyxy):
            if result.boxes.cls[i] not in self.__cls_indices[0:]:
                continue

            walking = True

            for retrieve in vehicle_retrieves:
                iou = cal_intersection_ratio(xyxy, result.boxes.xyxy[retrieve])

                if iou > self.__iou_threshold:
                    walking = False
                    break

            if walking:
                self.__id_set.add(result.boxes.id[i])

        return len(self.__id_set)


class PimDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            iou_threshold: float,
            duration_threshold: float,
            fps: float
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__polygon = Polygon(det_zone)
        self.__iou_threshold = iou_threshold
        self.__fps = fps
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__result_buffer = deque(maxlen=self.__duration_frame_num)
        self.__frame_buffer = deque(maxlen=self.__duration_frame_num)
        self.__pim_counts = {}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray,
    ):
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

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
                piming = self.__polygon.contains(Point((xywh[0], xyxy[3])))  # bottom mid

            ret[idx] = piming

            update_counts(
                piming,
                True,
                idx,
                self.__pim_counts,
                self.__output_countdowns,
                self.__duration_frame_num,
                1  # image instead of video
            )

        for idx in list(self.__output_countdowns.keys()):
            del self.__output_countdowns[idx]

            if result.boxes.id is None:
                continue

            retrieve = np.where(result.boxes.id == idx)[0]

            if retrieve.shape[0] >= 1:
                frame_copy = frame.copy()
                list_idx = retrieve[0]

                xyxy = result.boxes.xyxy[list_idx]

                cv2.rectangle(
                    frame_copy,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 0, 255),
                    thickness=2
                )

                cv2.imwrite(increment_path(f'runs/pim{str(int(idx)).zfill(7)}.png'), frame_copy)

        return ret


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
            result: Results,
            # frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray,
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
    ) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
        result = deepcopy(self.__result_buffer[-1])
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
                (int(xyxy[2]), int(xyxy[3])),
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

                dest = generate_hash20(f'parking{idx}{time()}')
                dests.append(dest)

                generate_video(
                    f'{output_dir}/{dest}.mp4',
                    idx,
                    self.__result_buffer,
                    self.__fps,
                    self.__video_frame_num
                )

        return dests


class WrongwayDetector:
    def __init__(
            self,
            cls_indices: list,
            det_zone: list,
            delta_second: float,
            duration_threshold: float,
            video_length: float,
            fps: float,
            correct_way='up'
    ):
        self.__id_set = set()
        self.__cls_indices = cls_indices
        self.__polygon = Polygon(det_zone)
        self.__fps = fps
        self.__correct_way = correct_way
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__frame_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__wrongway_counts = {}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray,
    ) -> dict[int, dict]:
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: {'vector': [0., 0.], 'wrongway': False} for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not self.__polygon.contains(center):
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
                    motion_vector /= norm
                    motion_vector = list(motion_vector)
                    break

            wrongway = self.__is_wrongway(motion_vector)
            ret[idx] = {'vector': motion_vector, 'wrongway': wrongway}

            update_counts(
                wrongway,
                True,
                idx,
                self.__wrongway_counts,
                self.__output_countdowns,
                self.__duration_frame_num,
                self.__video_frame_num
            )

        generate_videos_respectively(
            f'runs/wrongway',
            self.__result_buffer,
            self.__frame_buffer,
            self.__fps,
            self.__video_frame_num,
            self.__output_countdowns
        )

        return ret

    def __is_wrongway(self, motion_vector: list[float]) -> bool:
        if self.__correct_way == 'up':
            return motion_vector[1] > 0
        elif self.__correct_way == 'down':
            return motion_vector[1] < 0
        elif self.__correct_way == 'left':
            return motion_vector[0] < 0
        elif self.__correct_way == 'right':
            return motion_vector[0] > 0
        else:
            return False


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
        self.__polygon = Polygon(det_zone)
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__frame_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__solid_lines = [LineString(solid_line) for solid_line in solid_lines]
        self.__lanechange_counts = {}  # {id: count}
        self.__output_countdowns = {}

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
    ) -> dict[int, bool]:
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

        if result.boxes.id is None:
            return {}

        if len(self.__result_buffer) == 0:
            return {idx: False for idx in result.boxes.id}

        ret = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not self.__polygon.contains(center):
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

            update_counts(
                lanechange,
                True,
                idx,
                self.__lanechange_counts,
                self.__output_countdowns,
                self.__duration_frame_num,
                self.__video_frame_num
            )

        generate_videos_respectively(
            f'runs/lanechange',
            self.__result_buffer,
            self.__frame_buffer,
            self.__fps,
            self.__video_frame_num,
            self.__output_countdowns
        )

        return ret


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
        self.__polygon = Polygon(det_zone)
        self.__fps = fps
        self.__delta_frame_num = max(round(delta_second * fps), 1)
        self.__duration_frame_num = max(round(duration_threshold * fps), 1)
        self.__video_frame_num = max(round(video_length * fps), 1)
        self.__result_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__frame_buffer = deque(
            maxlen=max(self.__delta_frame_num, self.__duration_frame_num, self.__video_frame_num))
        self.__velocity_buffer = deque(
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

        self._homography_matrix = cal_homography_matrix(det_zone, [rv0, rv1, rv2, rv3])

    def update(
            self,
            result: Results,
            frame: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
    ) -> dict[int, float]:
        self.__result_buffer.append(result)
        self.__frame_buffer.append(frame)

        if result.boxes.id is None:
            self.__velocity_buffer.append({})
            return {}

        if len(self.__result_buffer) == 0:
            self.__velocity_buffer.append({idx: f'0.0 km/h' for idx in result.boxes.id})
            return {idx: 0 for idx in result.boxes.id}

        ret = {}
        texts = {}
        for i, idx in enumerate(result.boxes.id):
            center = Point(result.boxes.xywh[i][:2])
            if result.boxes.cls[i] not in self.__cls_indices or not self.__polygon.contains(center):
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
                    repro_points = reproject(points, self._homography_matrix)
                    distance = cal_euclidean_distance(repro_points[0], repro_points[1])
                    velocity = distance / ((len(self.__result_buffer) - j) / self.__fps) * 3.6
                    speeding = velocity >= self.__speed_threshold

                    break

            ret[idx] = speeding
            texts[idx] = f'{velocity:.3f} km/h'

            update_counts(
                speeding,
                True,
                idx,
                self.__velocity_counts,
                self.__output_countdowns,
                self.__duration_frame_num,
                self.__video_frame_num
            )

        generate_videos_respectively(
            f'runs/speeding',
            self.__result_buffer,
            self.__frame_buffer,
            self.__fps,
            self.__video_frame_num,
            self.__output_countdowns,
            self.__velocity_buffer
        )

        self.__velocity_buffer.append(texts)

        return ret
