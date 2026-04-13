"""
试管映射器模块
将检测到的液滴轮廓映射到网格位置
"""

import cv2
import numpy as np


class TubeMapper:
    """试管位置映射器"""

    ROW_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def __init__(self, rows=8, cols=12):
        """
        初始化映射器

        Args:
            rows: 行数
            cols: 列数
        """
        self.rows = rows
        self.cols = cols
        self.total_tubes = rows * cols

    def map_contours(self, contours, row_bounds, col_bounds):
        """
        将轮廓映射到网格位置

        Args:
            contours: 轮廓列表
            row_bounds: 行边界坐标
            col_bounds: 列边界坐标

        Returns:
            list: 2D数组，每个元素为tube信息字典
        """
        # 初始化tube矩阵（全部标记为empty）
        tube_matrix = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                row_data.append(
                    {
                        "row": self.ROW_LABELS[r],
                        "column": c + 1,
                        "status": "empty",
                        "droplet_area": None,
                        "centroid": None,
                    }
                )
            tube_matrix.append(row_data)

        # 计算每个轮廓的重心并分配位置
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(cnt)

                # 找最近的行和列
                row_idx = min(
                    range(len(row_bounds)), key=lambda i: abs(cy - row_bounds[i])
                )
                col_idx = min(
                    range(len(col_bounds)), key=lambda i: abs(cx - col_bounds[i])
                )

                # 更新tube状态
                tube_matrix[row_idx][col_idx]["status"] = "filled"
                tube_matrix[row_idx][col_idx]["droplet_area"] = area
                tube_matrix[row_idx][col_idx]["centroid"] = {"x": cx, "y": cy}

        return tube_matrix

    def get_statistics(self, tube_matrix):
        """
        统计填充情况

        Args:
            tube_matrix: tube矩阵

        Returns:
            dict: 统计信息
        """
        filled_count = sum(
            1 for row in tube_matrix for tube in row if tube["status"] == "filled"
        )
        empty_count = self.total_tubes - filled_count
        fill_rate = (filled_count / self.total_tubes) * 100

        return {
            "total_tubes": self.total_tubes,
            "filled_count": filled_count,
            "empty_count": empty_count,
            "fill_rate": fill_rate,
        }
