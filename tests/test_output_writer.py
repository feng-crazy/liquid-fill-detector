"""
OutputWriter测试模块
"""

import pytest
import os
import json
import tempfile
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
from lib.output_writer import OutputWriter


def test_output_writer_init():
    """测试OutputWriter初始化"""
    writer = OutputWriter(output_dir="output/test")
    assert writer.output_dir == "output/test"


def test_output_writer_save_json():
    """测试JSON保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OutputWriter(output_dir=tmpdir)

        test_data = {
            "plate_info": {"type": "test"},
            "tubes": [{"row": "A", "column": 1, "status": "filled"}],
        }

        writer.save_json(test_data, "result.json")

        # 验证文件存在
        filepath = os.path.join(tmpdir, "result.json")
        assert os.path.exists(filepath)

        # 验证内容
        with open(filepath) as f:
            loaded = json.load(f)
            assert loaded == test_data


def test_output_writer_save_image():
    """测试图像保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OutputWriter(output_dir=tmpdir)

        # 创建测试图像
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (255, 0, 0)  # 红色

        writer.save_image(test_img, "test.jpg")

        filepath = os.path.join(tmpdir, "test.jpg")
        assert os.path.exists(filepath)

        # 验证图像内容
        loaded = cv2.imread(filepath)
        assert loaded is not None
        assert loaded.shape == test_img.shape
