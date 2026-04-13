"""
径向微孔板检测器
使用Hough圆检测识别径向/扇形布局的微孔板
"""

import cv2
import numpy as np


class RadialDetector:
    """径向微孔板检测器"""

    def __init__(self, rows=8, cols=12):
        self.rows = rows
        self.cols = cols
        self.total_tubes = rows * cols

    def detect_wells(self, roi_image):
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.dilate(thresh, kernel, iterations=2)
        morphed = cv2.erode(morphed, kernel, iterations=1)

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 2000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    valid_contours.append((cx, cy, area, cnt))

        return valid_contours

    def filter_wells(self, circles, roi_image):
        """
        过滤检测到的圆，只保留有效的孔位

        Args:
            circles: 检测到的圆列表
            roi_image: ROI图像

        Returns:
            list: 过滤后的圆列表
        """
        if len(circles) < self.total_tubes * 0.8:
            return circles

        h, w = roi_image.shape[:2]

        valid_circles = []
        for x, y, r in circles:
            if 0 < x < w and 0 < y < h:
                if r > 5:
                    valid_circles.append((x, y, r))

        return valid_circles

    def map_to_grid(self, contours):
        if len(contours) == 0:
            return {}

        contours_sorted = sorted(contours, key=lambda c: (c[1], c[0]))

        positions = {}
        for i, contour_data in enumerate(contours_sorted):
            if i < self.total_tubes:
                cx, cy, area, cnt = contour_data
                row_idx = i // self.cols
                col_idx = i % self.cols
                positions[(row_idx, col_idx)] = contour_data

        return positions

    def check_fill_status(self, contour_data, roi_image):
        cx, cy, area, cnt = contour_data
        return "filled"

    def detect_plate(self, roi_image):
        contours = self.detect_wells(roi_image)
        positions = self.map_to_grid(contours)

        tube_matrix = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                pos_key = (r, c)
                if pos_key in positions:
                    contour_data = positions[pos_key]
                    cx, cy, area = contour_data[:3]
                    status = self.check_fill_status(contour_data, roi_image)
                    row_data.append(
                        {
                            "row": chr(ord("A") + r),
                            "column": c + 1,
                            "status": status,
                            "centroid": {"x": cx, "y": cy},
                            "droplet_area": area,
                        }
                    )
                else:
                    row_data.append(
                        {
                            "row": chr(ord("A") + r),
                            "column": c + 1,
                            "status": "empty",
                            "centroid": None,
                            "droplet_area": None,
                        }
                    )
            tube_matrix.append(row_data)

        filled_count = sum(
            1 for row in tube_matrix for tube in row if tube["status"] == "filled"
        )
        empty_count = self.total_tubes - filled_count
        fill_rate = (filled_count / self.total_tubes) * 100

        return {
            "tube_matrix": tube_matrix,
            "contours": contours,
            "stats": {
                "total_tubes": self.total_tubes,
                "filled_count": filled_count,
                "empty_count": empty_count,
                "fill_rate": fill_rate,
            },
        }
