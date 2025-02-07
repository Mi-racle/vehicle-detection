import math
from enum import unique, IntEnum
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

    iou = intersection_area / box1_area if box1_area > 0 else 0
    return iou


@unique
class VehicleSize(IntEnum):
    LARGE = 0
    MEDIUM = 1
    SMALL = 2


@unique
class VehicleState(IntEnum):
    UNKNOWN = 0
    MOVING = 1
    LEGALLY_PARKED = 2
    ILLEGALLY_PARKED = 3
