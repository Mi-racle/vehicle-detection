import math
from collections import deque
from enum import unique, IntEnum
from typing import Sequence, Any

import cv2
import numpy as np
import sympy as sp
from cv2 import Mat
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from ultralytics.engine.results import Results


def cal_intersection_points(center_a: Sequence[float], center_b: Sequence[float], r1: float, r2: float) -> list:
    x, y = sp.symbols('x y')

    x1, y1, *_ = center_a
    circle1 = (x - x1) ** 2 + (y - y1) ** 2 - r1 ** 2

    x2, y2, *_ = center_b
    circle2 = (x - x2) ** 2 + (y - y2) ** 2 - r2 ** 2

    return sp.solve([circle1, circle2], (x, y))


def cal_euclidean_distance(a: Sequence[float], b: Sequence[float]):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def cal_homography_matrix(src_points: list | np.ndarray, dst_points: list | np.ndarray) -> Mat | np.ndarray[Any, np.dtype] | np.ndarray:
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    homography_matrix, status = cv2.findHomography(src_points, dst_points)

    return homography_matrix


def reproject(points: list | np.ndarray, matrix: Mat | np.ndarray[Any, np.dtype] | np.ndarray) -> np.ndarray:
    """
    points: Ndarray, [[x, y], ...]
    matrix: Ndarray, 3 x 3
    """
    points = np.array(points)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    mapped_points_homogeneous = np.dot(matrix, points_homogeneous.T).T
    mapped_points = mapped_points_homogeneous[:, :2] / mapped_points_homogeneous[:, 2][:, np.newaxis]

    return mapped_points


def is_between(line: LineString, lower_line: LineString, upper_line: LineString):
    l_start, l_end = line.coords[0], line.coords[-1]

    for p in [l_start, l_end]:
        point = Point(p)
        y1 = lower_line.interpolate(lower_line.project(point)).y
        y2 = upper_line.interpolate(upper_line.project(point)).y

        if not min(y1, y2) < point.y <= max(y1, y2):
            return False

    return True


def cal_int_ratio(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    iou = intersection_area / box1_area if box1_area > 0 else 0
    return iou


def cal_velocity(p1: Sequence, p2: Sequence, dt=1):
    return (np.array(p2) - np.array(p1)) / dt


def cal_angle(p1: Sequence, p2: Sequence, p3: Sequence):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def is_walking(results: Sequence, idx: int, threshold=3):
    if len(results) < 2:
        return False

    velocities = []
    knee_angles = []

    for i in range(len(results) - 1):
        result_prev = results[i]
        result_curr = results[i + 1]

        if result_prev.boxes.id is None or result_curr.boxes.id is None:
            continue

        retrieve_prev = np.where(result_prev.boxes.id == idx)[0]
        retrieve_curr = np.where(result_curr.boxes.id == idx)[0]

        if retrieve_prev.shape[0] < 1 or retrieve_curr.shape[0] < 1:
            continue

        retrieve_prev = retrieve_prev[0]
        retrieve_curr = retrieve_curr[0]

        keypoints_prev = result_prev.keypoints.xy[retrieve_prev]
        keypoints_curr = result_curr.keypoints.xy[retrieve_curr]

        # 计算脚踝速度
        left_ankle_vel = cal_velocity(keypoints_prev[15], keypoints_curr[15])
        right_ankle_vel = cal_velocity(keypoints_prev[16], keypoints_curr[16])
        velocities.append((left_ankle_vel, right_ankle_vel))

        # 计算膝盖角度变化
        left_knee_angle = cal_angle(
            keypoints_curr[11], keypoints_curr[13], keypoints_curr[15]
        )
        right_knee_angle = cal_angle(
            keypoints_curr[12], keypoints_curr[14], keypoints_curr[16]
        )
        knee_angles.append((left_knee_angle, right_knee_angle))

    # 判断脚踝速度是否具有明显交替运动
    alternating_movement = sum(
        abs(v[0][1] - v[1][1]) > threshold for v in velocities
    ) > len(velocities) // 2

    # 判断膝盖角度是否有规律变化
    angle_changes = sum(
        abs(a[0] - a[1]) > threshold for a in knee_angles
    ) > len(knee_angles) // 2

    return alternating_movement and angle_changes


@unique
class VehicleSize(IntEnum):
    LARGE = 0
    MEDIUM = 1
    SMALL = 2
