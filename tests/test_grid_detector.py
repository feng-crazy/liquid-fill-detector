import pytest
import numpy as np
import cv2
import sys

sys.path.insert(0, ".")
from lib.grid_detector import GridDetector


def test_grid_detector_init():
    """测试GridDetector初始化"""
    detector = GridDetector(rows=8, cols=12)
    assert detector.rows == 8
    assert detector.cols == 12


def test_grid_detector_find_grid():
    """测试网格检测"""
    detector = GridDetector(rows=8, cols=12)

    test_img = np.zeros((400, 600), dtype=np.uint8)
    row_spacing = 400 // 8
    col_spacing = 600 // 12

    for i in range(8):
        for j in range(12):
            y = i * row_spacing + row_spacing // 2
            x = j * col_spacing + col_spacing // 2
            cv2.circle(test_img, (x, y), 5, 255, -1)

    row_bounds, col_bounds = detector.find_grid(test_img)

    assert len(row_bounds) == 8
    assert len(col_bounds) == 12


def test_grid_detector_assign_position():
    """测试行列位置分配"""
    detector = GridDetector(rows=8, cols=12)

    row_bounds = [25, 75, 125, 175, 225, 275, 325, 375]
    col_bounds = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575]

    row_idx, col_idx = detector.assign_position((300, 200), row_bounds, col_bounds)

    assert 0 <= row_idx < 8
    assert 0 <= col_idx < 12


def test_grid_detector_peak_distribution():
    """测试峰值分布正确性"""
    detector = GridDetector(rows=8, cols=12)

    test_proj = np.zeros(400)
    peak_positions = [20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 380]
    for pos in peak_positions:
        test_proj[pos] = 1000

    bounds = detector._find_boundaries(test_proj, 8, 400)

    assert len(bounds) == 8
    assert bounds[0] >= 40
    assert bounds[-1] >= 200
    assert len(bounds) == len(set(bounds)), (
        "Duplicate peaks detected - should have unique positions"
    )


def test_grid_detector_no_duplicate_selection():
    """测试不会重复选择峰值"""
    detector = GridDetector(rows=8, cols=12)

    test_proj = np.zeros(400)
    peak_positions = [40, 50, 60, 70, 80, 200, 300, 350, 380]
    for pos in peak_positions:
        test_proj[pos] = 1000

    bounds = detector._find_boundaries(test_proj, 8, 400)

    assert len(bounds) == 8
    assert len(bounds) == len(set(bounds)), "Duplicate peaks should not occur"
