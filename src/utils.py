import glob
import math
import re
from pathlib import Path
from threading import Thread
from typing import Sequence, Any

import cv2
import numpy as np
import sympy as sp
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point


def cal_intersection_points(
        center_a: Sequence[float],
        center_b: Sequence[float],
        r1: float,
        r2: float
) -> list:
    x, y = sp.symbols('x y')

    x1, y1, *_ = center_a
    circle1 = (x - x1) ** 2 + (y - y1) ** 2 - r1 ** 2

    x2, y2, *_ = center_b
    circle2 = (x - x2) ** 2 + (y - y2) ** 2 - r2 ** 2

    return sp.solve([circle1, circle2], (x, y))


def cal_euclidean_distance(
        a: Sequence[float],
        b: Sequence[float]
):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def cal_homography_matrix(
        src_points: list | np.ndarray,
        dst_points: list | np.ndarray
) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray:
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    homography_matrix, status = cv2.findHomography(src_points, dst_points)

    return homography_matrix


def reproject(
        points: list | np.ndarray,
        matrix: cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray
) -> np.ndarray:
    """
    points: Ndarray, [[x, y], ...]
    matrix: Ndarray, 3 x 3
    """
    points = np.array(points)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    mapped_points_homogeneous = np.dot(matrix, points_homogeneous.T).T
    mapped_points = mapped_points_homogeneous[:, :2] / mapped_points_homogeneous[:, 2][:, np.newaxis]

    return mapped_points


def is_between(
        line: LineString,
        lower_line: LineString,
        upper_line: LineString
) -> bool:
    l_start, l_end = line.coords[0], line.coords[-1]

    for p in [l_start, l_end]:
        point = Point(p)
        y1 = lower_line.interpolate(lower_line.project(point)).y
        y2 = upper_line.interpolate(upper_line.project(point)).y

        if not min(y1, y2) < point.y <= max(y1, y2):
            return False

    return True


def cal_intersection_ratio(
        box1: Sequence,
        box2: Sequence
) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    return intersection_area / box1_area if box1_area > 0 else 0


def increment_path(dst_path: str, exist_ok=False, sep='', mkdir=False):
    """Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc."""
    dst_path = Path(dst_path)  # os-agnostic

    if dst_path.exists() and not exist_ok:
        suffix = dst_path.suffix
        dst_path = dst_path.with_suffix('')
        dirs = glob.glob(f"{dst_path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % dst_path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 1  # increment number
        dst_path = Path(f"{dst_path}{sep}{n}{suffix}")  # update path

    _dir = dst_path if dst_path.suffix == '' else dst_path.parent  # directory

    if not _dir.exists() and mkdir:
        _dir.mkdir(parents=True, exist_ok=True)  # make directory

    return str(dst_path)


def generate_video(
        dst_name: str,
        frames: Sequence[cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray],
        fps: float
):
    if not frames:
        return

    cap_out = cv2.VideoWriter(
        increment_path(dst_name),
        cv2.VideoWriter.fourcc(*'mp4v'),
        fps,
        (frames[0].shape[1], frames[0].shape[0])
    )

    for frame in frames:
        cap_out.write(frame)

    cap_out.release()


def generate_videos(
        dst_name: str,
        result_buffer: Sequence,
        frame_buffer: Sequence[cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray],
        fps: float,
        frame_num: int,
        countdowns: dict,
        text_buffer: Sequence[dict] | None = None
) -> dict:
    for idx in list(countdowns.keys()):
        countdowns[idx] -= 1

        if countdowns[idx] <= 0:
            del countdowns[idx]

            frames_to_write = []
            for k, buffered_frame in enumerate(list(frame_buffer)[-frame_num:]):
                frame_copy = buffered_frame.copy()
                buffered_result = result_buffer[k - min(frame_num, len(result_buffer))]

                if buffered_result.boxes.id is None:
                    continue

                retrieve = np.where(buffered_result.boxes.id == idx)[0]

                if retrieve.shape[0] >= 1:
                    list_idx = retrieve[0]

                    xyxy = buffered_result.boxes.xyxy[list_idx]

                    cv2.rectangle(
                        frame_copy,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255),
                        thickness=2
                    )

                    # TODO bug
                    texts = text_buffer[k]
                    if texts:
                        cv2.putText(
                            frame_copy,
                            texts[idx],
                            (int(xyxy[2]), int(xyxy[1] + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .4,
                            (0, 0, 255)
                        )

                frames_to_write.append(frame_copy)

            Thread(
                target=generate_video,
                args=(dst_name + f'{str(int(idx)).zfill(7)}.mp4', frames_to_write, fps),
                daemon=True
            ).start()

    return countdowns


def update_counts(
        value: Any,
        target_value: Any,
        idx,
        counts: dict,
        countdowns: dict,
        duration_frame_num: int,
        video_frame_num: int
):
    if value == target_value:
        if idx in counts:
            counts[idx] += 1
        elif idx not in countdowns:
            counts[idx] = 1
    
        if idx in counts and counts[idx] >= duration_frame_num:
            countdowns[idx] = video_frame_num // 2
            del counts[idx]
    
    else:
        if idx in counts:
            del counts[idx]
