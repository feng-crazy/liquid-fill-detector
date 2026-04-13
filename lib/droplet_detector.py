"""
液滴检测器模块
用于检测黑色液滴并提取轮廓
"""

import cv2
import numpy as np


class DropletDetector:
    """黑色液滴检测器"""

    def __init__(
        self,
        threshold_value=60,
        min_area=20,
        max_area=2000,
        use_adaptive=False,
        adaptive_block_size=15,
        adaptive_c=5,
        use_opening_preprocess=True,
        opening_kernel_size=3,
    ):
        """
        初始化检测器

        Args:
            threshold_value: 灰度阈值，低于此值视为黑色液滴（固定阈值模式）
            min_area: 最小轮廓面积，过滤噪点
            max_area: 最大轮廓面积，过滤大块干扰
            use_adaptive: 是否使用自适应阈值（默认False，固定阈值更适合此场景）
            adaptive_block_size: 自适应阈值邻域大小（必须为奇数）
            adaptive_c: 自适应阈值常数
            use_opening_preprocess: 是否使用形态学开运算预处理去除反光噪点
            opening_kernel_size: 开运算核大小（默认3，去除细小反光点）
        """
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area
        self.use_adaptive = use_adaptive
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.use_opening_preprocess = use_opening_preprocess
        self.opening_kernel_size = opening_kernel_size

    def detect(self, gray_image):
        """
        检测液滴轮廓

        Args:
            gray_image: 灰度图像

        Returns:
            tuple: (valid_contours, thresh, morphed) 有效轮廓列表、阈值图、形态学处理图
        """
        processed = gray_image.copy()

        # 形态学开运算预处理：去除反光噪点（高亮细小白点）
        if self.use_opening_preprocess:
            opening_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.opening_kernel_size, self.opening_kernel_size),
            )
            processed = cv2.morphologyEx(
                processed, cv2.MORPH_OPEN, opening_kernel, iterations=1
            )

        # 阈值分割
        if self.use_adaptive:
            thresh = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.adaptive_block_size,
                self.adaptive_c,
            )
        else:
            _, thresh = cv2.threshold(
                processed, self.threshold_value, 255, cv2.THRESH_BINARY_INV
            )

        # 形态学处理：椭圆核，膨胀2次+腐蚀1次（填充小孔洞，连接断裂边缘）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.dilate(thresh, kernel, iterations=2)
        morphed = cv2.erode(morphed, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 过滤轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                valid_contours.append(cnt)

        return valid_contours, thresh, morphed
