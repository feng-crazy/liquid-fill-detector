"""
输出写入器模块
生成JSON报告和可视化图像
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime


class OutputWriter:
    """输出结果写入器"""

    def __init__(self, output_dir):
        """
        初始化写入器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data, filename):
        """
        保存JSON文件

        Args:
            data: 数据字典
            filename: 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_image(self, image, filename):
        """
        保存图像

        Args:
            image: 图像数组
            filename: 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image)

    def generate_result_json(self, tube_matrix, plate_info, timestamp):
        """
        生成完整结果JSON

        Args:
            tube_matrix: tube状态矩阵
            plate_info: 板信息字典
            timestamp: 时间戳

        Returns:
            dict: 结果数据
        """
        tubes_flat = [tube for row in tube_matrix for tube in row]

        return {"plate_info": plate_info, "tubes": tubes_flat, "timestamp": timestamp}

    def generate_stats_json(self, stats, verdict, processing_time_ms):
        """
        生成统计JSON

        Args:
            stats: 统计信息字典
            verdict: 判定结果
            processing_time_ms: 处理时间

        Returns:
            dict: 统计数据
        """
        return {**stats, "verdict": verdict, "processing_time_ms": processing_time_ms}

    def draw_annotated_image(self, roi_image, tube_matrix, row_bounds, col_bounds):
        """
        绘制标注图像

        Args:
            roi_image: ROI彩色图像
            tube_matrix: tube状态矩阵
            row_bounds: 行边界
            col_bounds: 列边界

        Returns:
            ndarray: 标注后的图像
        """
        annotated = roi_image.copy()

        # 绘制网格线
        for y in row_bounds:
            cv2.line(annotated, (0, y), (annotated.shape[1], y), (100, 100, 100), 1)
        for x in col_bounds:
            cv2.line(annotated, (x, 0), (x, annotated.shape[0]), (100, 100, 100), 1)

        # 绘制tube状态
        for r, row in enumerate(tube_matrix):
            for c, tube in enumerate(row):
                centroid = tube.get("centroid")
                if centroid and isinstance(centroid, dict):
                    x, y = centroid.get("x", 0), centroid.get("y", 0)
                    status = tube.get("status", "empty")
                    color = (0, 255, 0) if status == "filled" else (0, 0, 255)
                    cv2.circle(annotated, (x, y), 8, color, -1)

                    # 标注行列标签
                    label = f"{tube.get('row', '?')}{tube.get('column', '?')}"
                    cv2.putText(
                        annotated,
                        label,
                        (x - 10, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        color,
                        1,
                    )

        # 绘制统计数据
        total = (
            len(tube_matrix) * len(tube_matrix[0])
            if tube_matrix and tube_matrix[0]
            else 0
        )
        filled = sum(
            1 for row in tube_matrix for tube in row if tube.get("status") == "filled"
        )
        empty = total - filled
        verdict_text = "PASS" if empty == 0 else f"FAIL ({empty} empty)"
        verdict_color = (0, 255, 0) if empty == 0 else (0, 0, 255)

        cv2.putText(
            annotated,
            f"Filled: {filled}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            verdict_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            verdict_color,
            2,
        )

        return annotated
