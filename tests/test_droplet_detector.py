"""
DropletDetector测试模块
"""

import pytest
import numpy as np
import cv2
import sys

sys.path.insert(0, ".")
from lib.droplet_detector import DropletDetector


def test_droplet_detector_init():
    """测试DropletDetector初始化"""
    detector = DropletDetector(threshold_value=60, min_area=20, max_area=2000)
    assert detector.threshold_value == 60
    assert detector.min_area == 20
    assert detector.max_area == 2000


def test_droplet_detector_default_config():
    """测试默认配置"""
    detector = DropletDetector()
    assert detector.threshold_value == 60
    assert detector.min_area == 20
    assert detector.max_area == 2000


def test_droplet_detector_detect():
    """测试检测方法"""
    detector = DropletDetector()

    # 创建测试图像：中心有一个黑色圆形
    test_img = np.ones((100, 100), dtype=np.uint8) * 200  # 浅灰背景
    cv2.circle(test_img, (50, 50), 10, 0, -1)  # 黑色圆

    contours, thresh, morphed = detector.detect(test_img)

    assert len(contours) == 1  # 应检测到1个轮廓
    assert thresh.shape == test_img.shape
    assert morphed.shape == test_img.shape


def test_droplet_detector_filter_noise():
    """测试噪点过滤"""
    detector = DropletDetector(min_area=50)

    # 创建测试图像：中心有大圆，周围有小噪点
    test_img = np.ones((100, 100), dtype=np.uint8) * 200
    cv2.circle(test_img, (50, 50), 10, 0, -1)  # 大圆
    cv2.circle(test_img, (20, 20), 2, 0, -1)  # 小噪点

    contours, _, _ = detector.detect(test_img)

    assert len(contours) == 1  # 只保留大圆
