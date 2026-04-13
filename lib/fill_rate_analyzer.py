"""
填充率分析器模块
基于网格单元的填充率计算和动态阈值判定
"""

import cv2
import numpy as np


class FillRateAnalyzer:
    """网格化填充率分析器"""

    def __init__(self, rows=8, cols=12, k_factor=2.0, min_fill_threshold=0.1):
        """
        初始化分析器

        Args:
            rows: 行数
            cols: 列数
            k_factor: 动态阈值系数，阈值 = μ - k*σ
            min_fill_threshold: 最小填充率阈值，低于此值必定为空
        """
        self.rows = rows
        self.cols = cols
        self.k_factor = k_factor
        self.min_fill_threshold = min_fill_threshold

    def calculate_fill_rates(self, binary_image, row_bounds, col_bounds):
        """
        计算每个网格单元的填充率

        Args:
            binary_image: 二值图像（白色为前景）
            row_bounds: 行边界坐标列表
            col_bounds: 列边界坐标列表

        Returns:
            numpy.ndarray: 填充率矩阵 (rows x cols)
        """
        h, w = binary_image.shape
        fill_rates = np.zeros((self.rows, self.cols))

        for r in range(self.rows):
            for c in range(self.cols):
                row_center = row_bounds[r]
                col_center = col_bounds[c]

                row_spacing = (
                    row_bounds[min(r + 1, self.rows - 1)] - row_bounds[max(r - 1, 0)]
                ) // 2
                col_spacing = (
                    col_bounds[min(c + 1, self.cols - 1)] - col_bounds[max(c - 1, 0)]
                ) // 2

                if r == 0:
                    row_spacing = (row_bounds[1] - row_bounds[0]) // 2
                elif r == self.rows - 1:
                    row_spacing = (row_bounds[-1] - row_bounds[-2]) // 2
                else:
                    row_spacing = (row_bounds[r + 1] - row_bounds[r - 1]) // 2

                if c == 0:
                    col_spacing = (col_bounds[1] - col_bounds[0]) // 2
                elif c == self.cols - 1:
                    col_spacing = (col_bounds[-1] - col_bounds[-2]) // 2
                else:
                    col_spacing = (col_bounds[c + 1] - col_bounds[c - 1]) // 2

                y_start = max(0, row_center - row_spacing // 2)
                y_end = min(h, row_center + row_spacing // 2)
                x_start = max(0, col_center - col_spacing // 2)
                x_end = min(w, col_center + col_spacing // 2)

                cell = binary_image[y_start:y_end, x_start:x_end]
                total_pixels = cell.size
                white_pixels = np.sum(cell > 0)
                fill_rates[r, c] = (
                    white_pixels / total_pixels if total_pixels > 0 else 0
                )

        return fill_rates

    def calculate_dynamic_threshold(self, fill_rates):
        """
        计算动态阈值

        Args:
            fill_rates: 填充率矩阵

        Returns:
            float: 动态阈值
        """
        mean_fill = np.mean(fill_rates)
        std_fill = np.std(fill_rates)
        dynamic_threshold = mean_fill - self.k_factor * std_fill
        threshold = max(dynamic_threshold, self.min_fill_threshold)
        max_threshold = mean_fill * 0.5
        if threshold > max_threshold:
            threshold = max(max_threshold, self.min_fill_threshold)
        return threshold

    def classify_cells(self, fill_rates, threshold=None):
        """
        分类网格单元为填充或空

        Args:
            fill_rates: 填充率矩阵
            threshold: 阈值（可选，不提供则使用动态阈值）

        Returns:
            numpy.ndarray: 状态矩阵 (rows x cols), True为填充，False为空
        """
        if threshold is None:
            threshold = self.calculate_dynamic_threshold(fill_rates)

        return fill_rates >= threshold

    def analyze(self, binary_image, row_bounds, col_bounds):
        """
        执行完整分析

        Args:
            binary_image: 二值图像
            row_bounds: 行边界
            col_bounds: 列边界

        Returns:
            dict: 分析结果，包含填充率矩阵、状态矩阵、统计信息
        """
        fill_rates = self.calculate_fill_rates(binary_image, row_bounds, col_bounds)
        dynamic_threshold = self.calculate_dynamic_threshold(fill_rates)
        filled_matrix = self.classify_cells(fill_rates, dynamic_threshold)

        filled_count = np.sum(filled_matrix)
        empty_count = self.rows * self.cols - filled_count
        fill_rate = (filled_count / (self.rows * self.cols)) * 100

        return {
            "fill_rates": fill_rates,
            "filled_matrix": filled_matrix,
            "dynamic_threshold": dynamic_threshold,
            "mean_fill_rate": np.mean(fill_rates),
            "std_fill_rate": np.std(fill_rates),
            "stats": {
                "total_tubes": self.rows * self.cols,
                "filled_count": int(filled_count),
                "empty_count": int(empty_count),
                "fill_rate": fill_rate,
            },
        }

    def get_cell_details(self, fill_rates, filled_matrix):
        """
        获取每个单元的详细信息

        Args:
            fill_rates: 填充率矩阵
            filled_matrix: 状态矩阵

        Returns:
            list: 单元详细信息列表
        """
        details = []
        for r in range(self.rows):
            for c in range(self.cols):
                details.append(
                    {
                        "row": chr(ord("A") + r),
                        "column": c + 1,
                        "fill_rate": fill_rates[r, c],
                        "status": "filled" if filled_matrix[r, c] else "empty",
                    }
                )
        return details
