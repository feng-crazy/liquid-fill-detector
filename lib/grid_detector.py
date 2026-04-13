"""
网格检测器模块
通过投影分析检测微孔板的行列结构
"""

import cv2
import numpy as np


class GridDetector:
    """网格结构检测器"""

    def __init__(self, rows=8, cols=12):
        """
        初始化检测器

        Args:
            rows: 行数（默认8行）
            cols: 列数（默认12列）
        """
        self.rows = rows
        self.cols = cols

    def find_grid(self, binary_image):
        """
        检测网格行列边界

        Args:
            binary_image: 二值图像（白色为液滴）

        Returns:
            tuple: (row_boundaries, col_boundaries) 行列边界坐标列表
        """
        h, w = binary_image.shape

        h_proj = np.sum(binary_image, axis=1)
        v_proj = np.sum(binary_image, axis=0)

        row_bounds = self._find_boundaries(h_proj, self.rows, h)
        col_bounds = self._find_boundaries(v_proj, self.cols, w)

        return row_bounds, col_bounds

    def _find_boundaries(self, projection, num_regions, total_length):
        """
        从投影曲线找到区域边界

        Args:
            projection: 投影数组
            num_regions: 区域数量
            total_length: 总长度

        Returns:
            list: 边界坐标列表（每个区域的中心位置）
        """
        if np.std(projection) < 10:
            spacing = total_length // num_regions
            return [spacing // 2 + i * spacing for i in range(num_regions)]

        peaks = []
        threshold = np.mean(projection) + np.std(projection)

        for i in range(1, len(projection) - 1):
            if (
                projection[i] > threshold
                and projection[i] > projection[i - 1]
                and projection[i] > projection[i + 1]
            ):
                peaks.append(i)

        if len(peaks) < num_regions:
            spacing = total_length // num_regions
            return [spacing // 2 + i * spacing for i in range(num_regions)]

        peaks = sorted(peaks)

        if len(peaks) > num_regions:
            spacing = total_length // (num_regions + 1)
            selected_peaks = []
            available_peaks = peaks.copy()
            for i in range(num_regions):
                ideal_pos = spacing + i * spacing
                nearest_peak = min(available_peaks, key=lambda p: abs(p - ideal_pos))
                selected_peaks.append(nearest_peak)
                available_peaks.remove(nearest_peak)
            peaks = selected_peaks

        return peaks

    def assign_position(self, centroid, row_bounds, col_bounds):
        """
        根据重心坐标分配行列位置

        Args:
            centroid: (x, y) 重心坐标
            row_bounds: 行边界列表
            col_bounds: 列边界列表

        Returns:
            tuple: (row_index, col_index) 行列索引（0-based）
        """
        x, y = centroid

        row_idx = min(range(len(row_bounds)), key=lambda i: abs(y - row_bounds[i]))
        col_idx = min(range(len(col_bounds)), key=lambda i: abs(x - col_bounds[i]))

        return row_idx, col_idx
