import pytest
import numpy as np
import cv2
import sys

sys.path.insert(0, ".")
from lib.tube_mapper import TubeMapper


def test_tube_mapper_init():
    """测试TubeMapper初始化"""
    mapper = TubeMapper(rows=8, cols=12)
    assert mapper.rows == 8
    assert mapper.cols == 12
    assert mapper.total_tubes == 96


def test_tube_mapper_map_contours():
    """测试轮廓映射"""
    mapper = TubeMapper(rows=8, cols=12)

    # 创建模拟轮廓（中心坐标）
    test_img = np.zeros((400, 600), dtype=np.uint8)

    row_spacing = 400 // 8
    col_spacing = 600 // 12

    for i in range(8):
        for j in range(12):
            y = i * row_spacing + row_spacing // 2
            x = j * col_spacing + col_spacing // 2
            cv2.circle(test_img, (x, y), 5, 255, -1)

    contours_found, _ = cv2.findContours(
        test_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    row_bounds = [row_spacing // 2 + i * row_spacing for i in range(8)]
    col_bounds = [col_spacing // 2 + j * col_spacing for j in range(12)]

    tube_matrix = mapper.map_contours(contours_found, row_bounds, col_bounds)

    assert len(tube_matrix) == 8  # 8行
    assert len(tube_matrix[0]) == 12  # 12列
    # 应该检测到所有96个tube
    filled_count = sum(
        1 for row in tube_matrix for tube in row if tube["status"] == "filled"
    )
    assert filled_count >= 90  # 允许少量误差


def test_tube_mapper_get_statistics():
    """测试统计功能"""
    mapper = TubeMapper(rows=8, cols=12)

    # 创建部分填充的矩阵
    tube_matrix = []
    for r in range(8):
        row_data = []
        for c in range(12):
            status = "filled" if (r + c) % 2 == 0 else "empty"
            row_data.append(
                {
                    "row": mapper.ROW_LABELS[r],
                    "column": c + 1,
                    "status": status,
                    "droplet_area": 100 if status == "filled" else None,
                    "centroid": {"x": 0, "y": 0} if status == "filled" else None,
                }
            )
        tube_matrix.append(row_data)

    stats = mapper.get_statistics(tube_matrix)

    assert stats["total_tubes"] == 96
    assert stats["filled_count"] == 48
    assert stats["empty_count"] == 48
    assert stats["fill_rate"] == 50.0
