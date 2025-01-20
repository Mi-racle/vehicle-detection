import math
from enum import unique, IntEnum
from typing import Sequence, Any

import cv2
import numpy as np
import sympy as sp
from cv2 import Mat


def cal_intersection_points(centre_a: Sequence[float], centre_b: Sequence[float], r1: float, r2: float) -> list:
    x, y = sp.symbols('x y')

    x1, y1, *_ = centre_a
    circle1 = (x - x1) ** 2 + (y - y1) ** 2 - r1 ** 2

    x2, y2, *_ = centre_b
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


@unique
class VehicleSize(IntEnum):
    LARGE = 0
    MEDIUM = 1
    SMALL = 2
