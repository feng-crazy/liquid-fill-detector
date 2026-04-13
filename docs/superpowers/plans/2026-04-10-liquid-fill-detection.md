# Liquid Fill Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a machine vision system to detect liquid fill defects in 96-well microplates with black liquid, outputting per-tube analysis and intermediate visualization images.

**Architecture:** Modular pipeline with 4 components: DropletDetector (threshold segmentation), GridDetector (projection analysis), TubeMapper (grid position assignment), OutputWriter (JSON + 8 intermediate images). Entry script orchestrates the flow.

**Tech Stack:** Python 3.13, OpenCV (cv2), NumPy, JSON standard library

---

## Task 1: Project Structure Setup

**Files:**
- Create: `lib/__init__.py`
- Create: `output/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create lib package directory**

```bash
mkdir -p lib tests
touch lib/__init__.py tests/__init__.py
touch output/.gitkeep
```

- [ ] **Step 2: Verify structure**

Run: `ls -la lib/ tests/ output/`
Expected: Empty directories with __init__.py files

- [ ] **Step 3: Commit**

```bash
git add lib/ tests/ output/.gitkeep
git commit -m "feat: setup project structure for detection module"
```

---

## Task 2: DropletDetector - Threshold Segmentation

**Files:**
- Create: `lib/droplet_detector.py`
- Create: `tests/test_droplet_detector.py`

- [ ] **Step 1: Write failing test for DropletDetector initialization**

```python
# tests/test_droplet_detector.py
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_droplet_detector.py::test_droplet_detector_init -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'lib.droplet_detector'"

- [ ] **Step 3: Write minimal DropletDetector class**

```python
# lib/droplet_detector.py
"""
液滴检测器模块
用于检测黑色液滴并提取轮廓
"""

import cv2
import numpy as np


class DropletDetector:
    """黑色液滴检测器"""
    
    def __init__(self, threshold_value=60, min_area=20, max_area=2000):
        """
        初始化检测器
        
        Args:
            threshold_value: 灰度阈值，低于此值视为黑色液滴
            min_area: 最小轮廓面积，过滤噪点
            max_area: 最大轮廓面积，过滤大块干扰
        """
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area
    
    def detect(self, gray_image):
        """
        检测液滴轮廓
        
        Args:
            gray_image: 灰度图像
        
        Returns:
            list: 有效轮廓列表
        """
        # 阈值分割
        _, thresh = cv2.threshold(gray_image, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学处理：椭圆核，膨胀2次+腐蚀1次
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.dilate(thresh, kernel, iterations=2)
        morphed = cv2.erode(morphed, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                valid_contours.append(cnt)
        
        return valid_contours, thresh, morphed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_droplet_detector.py::test_droplet_detector_init -v`
Expected: PASS

- [ ] **Step 5: Write test for detect method**

```python
# tests/test_droplet_detector.py (append)
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
    cv2.circle(test_img, (20, 20), 2, 0, -1)   # 小噪点
    
    contours, _, _ = detector.detect(test_img)
    
    assert len(contours) == 1  # 只保留大圆
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_droplet_detector.py -v`
Expected: 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add lib/droplet_detector.py tests/test_droplet_detector.py
git commit -m "feat: add DropletDetector with threshold segmentation"
```

---

## Task 3: GridDetector - Projection Analysis

**Files:**
- Create: `lib/grid_detector.py`
- Create: `tests/test_grid_detector.py`

- [ ] **Step 1: Write failing test for GridDetector**

```python
# tests/test_grid_detector.py
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')
from lib.grid_detector import GridDetector

def test_grid_detector_init():
    """测试GridDetector初始化"""
    detector = GridDetector(rows=8, cols=12)
    assert detector.rows == 8
    assert detector.cols == 12

def test_grid_detector_find_grid():
    """测试网格检测"""
    detector = GridDetector(rows=8, cols=12)
    
    # 创建模拟图像：8行12列的网格点
    test_img = np.zeros((400, 600), dtype=np.uint8)
    
    # 每行8个点，每列12个点
    row_spacing = 400 // 8
    col_spacing = 600 // 12
    
    for i in range(8):
        for j in range(12):
            y = i * row_spacing + row_spacing // 2
            x = j * col_spacing + col_spacing // 2
            cv2.circle(test_img, (x, y), 5, 255, -1)
    
    row_bounds, col_bounds = detector.find_grid(test_img)
    
    assert len(row_bounds) == 8  # 8行边界
    assert len(col_bounds) == 12  # 12列边界
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_grid_detector.py::test_grid_detector_init -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal GridDetector class**

```python
# lib/grid_detector.py
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
        
        # 水平投影：每行像素累加
        h_proj = np.sum(binary_image, axis=1)
        
        # 垂直投影：每列像素累加
        v_proj = np.sum(binary_image, axis=0)
        
        # 找峰值位置（液滴所在行列）
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
        # 如果投影太平坦，使用均匀分布
        if np.std(projection) < 10:
            spacing = total_length // num_regions
            return [spacing // 2 + i * spacing for i in range(num_regions)]
        
        # 找峰值
        peaks = []
        threshold = np.mean(projection) + np.std(projection)
        
        for i in range(1, len(projection) - 1):
            if projection[i] > threshold and projection[i] > projection[i-1] and projection[i] > projection[i+1]:
                peaks.append(i)
        
        # 如果找到的峰值数量不够，均匀补充
        if len(peaks) < num_regions:
            spacing = total_length // num_regions
            return [spacing // 2 + i * spacing for i in range(num_regions)]
        
        # 选取前num_regions个峰值
        peaks = sorted(peaks)[:num_regions]
        
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
        
        # 找最近的行
        row_idx = min(range(len(row_bounds)), key=lambda i: abs(y - row_bounds[i]))
        
        # 找最近的列
        col_idx = min(range(len(col_bounds)), key=lambda i: abs(x - col_bounds[i]))
        
        return row_idx, col_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_grid_detector.py::test_grid_detector_init -v`
Expected: PASS

- [ ] **Step 5: Write test for assign_position method**

```python
# tests/test_grid_detector.py (append)
import cv2

def test_grid_detector_assign_position():
    """测试行列位置分配"""
    detector = GridDetector(rows=8, cols=12)
    
    row_bounds = [25, 75, 125, 175, 225, 275, 325, 375]  # 8行
    col_bounds = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575]  # 12列
    
    # 测试中心位置
    row_idx, col_idx = detector.assign_position((300, 200), row_bounds, col_bounds)
    
    assert 0 <= row_idx < 8
    assert 0 <= col_idx < 12
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_grid_detector.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add lib/grid_detector.py tests/test_grid_detector.py
git commit -m "feat: add GridDetector with projection analysis"
```

---

## Task 4: TubeMapper - Grid Position Mapping

**Files:**
- Create: `lib/tube_mapper.py`
- Create: `tests/test_tube_mapper.py`

- [ ] **Step 1: Write failing test for TubeMapper**

```python
# tests/test_tube_mapper.py
import pytest
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')
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
    contours = []
    test_img = np.zeros((400, 600), dtype=np.uint8)
    
    row_spacing = 400 // 8
    col_spacing = 600 // 12
    
    for i in range(8):
        for j in range(12):
            y = i * row_spacing + row_spacing // 2
            x = j * col_spacing + col_spacing // 2
            cv2.circle(test_img, (x, y), 5, 255, -1)
    
    contours_found, _ = cv2.findContours(test_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    row_bounds = [row_spacing // 2 + i * row_spacing for i in range(8)]
    col_bounds = [col_spacing // 2 + j * col_spacing for j in range(12)]
    
    tube_matrix = mapper.map_contours(contours_found, row_bounds, col_bounds)
    
    assert len(tube_matrix) == 8  # 8行
    assert len(tube_matrix[0]) == 12  # 12列
    # 应该检测到所有96个tube
    filled_count = sum(1 for row in tube_matrix for tube in row if tube['status'] == 'filled')
    assert filled_count >= 90  # 允许少量误差
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_tube_mapper.py::test_tube_mapper_init -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal TubeMapper class**

```python
# lib/tube_mapper.py
"""
试管映射器模块
将检测到的液滴轮廓映射到网格位置
"""

import cv2
import numpy as np


class TubeMapper:
    """试管位置映射器"""
    
    ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
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
                row_data.append({
                    'row': self.ROW_LABELS[r],
                    'column': c + 1,
                    'status': 'empty',
                    'droplet_area': None,
                    'centroid': None
                })
            tube_matrix.append(row_data)
        
        # 计算每个轮廓的重心并分配位置
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(cnt)
                
                # 找最近的行和列
                row_idx = min(range(len(row_bounds)), key=lambda i: abs(cy - row_bounds[i]))
                col_idx = min(range(len(col_bounds)), key=lambda i: abs(cx - col_bounds[i]))
                
                # 更新tube状态
                tube_matrix[row_idx][col_idx]['status'] = 'filled'
                tube_matrix[row_idx][col_idx]['droplet_area'] = area
                tube_matrix[row_idx][col_idx]['centroid'] = {'x': cx, 'y': cy}
        
        return tube_matrix
    
    def get_statistics(self, tube_matrix):
        """
        统计填充情况
        
        Args:
            tube_matrix: tube矩阵
        
        Returns:
            dict: 统计信息
        """
        filled_count = sum(1 for row in tube_matrix for tube in row if tube['status'] == 'filled')
        empty_count = self.total_tubes - filled_count
        fill_rate = (filled_count / self.total_tubes) * 100
        
        return {
            'total_tubes': self.total_tubes,
            'filled_count': filled_count,
            'empty_count': empty_count,
            'fill_rate': fill_rate
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_tube_mapper.py::test_tube_mapper_init -v`
Expected: PASS

- [ ] **Step 5: Write test for get_statistics**

```python
# tests/test_tube_mapper.py (append)
def test_tube_mapper_get_statistics():
    """测试统计功能"""
    mapper = TubeMapper(rows=8, cols=12)
    
    # 创建部分填充的矩阵
    tube_matrix = []
    for r in range(8):
        row_data = []
        for c in range(12):
            status = 'filled' if (r + c) % 2 == 0 else 'empty'
            row_data.append({
                'row': mapper.ROW_LABELS[r],
                'column': c + 1,
                'status': status,
                'droplet_area': 100 if status == 'filled' else None,
                'centroid': {'x': 0, 'y': 0} if status == 'filled' else None
            })
        tube_matrix.append(row_data)
    
    stats = mapper.get_statistics(tube_matrix)
    
    assert stats['total_tubes'] == 96
    assert stats['filled_count'] == 48
    assert stats['empty_count'] == 48
    assert stats['fill_rate'] == 50.0
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_tube_mapper.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add lib/tube_mapper.py tests/test_tube_mapper.py
git commit -m "feat: add TubeMapper for grid position assignment"
```

---

## Task 5: OutputWriter - JSON and Image Output

**Files:**
- Create: `lib/output_writer.py`
- Create: `tests/test_output_writer.py`

- [ ] **Step 1: Write failing test for OutputWriter**

```python
# tests/test_output_writer.py
import pytest
import os
import json
import tempfile
import sys
sys.path.insert(0, '.')
from lib.output_writer import OutputWriter

def test_output_writer_init():
    """测试OutputWriter初始化"""
    writer = OutputWriter(output_dir='output/test')
    assert writer.output_dir == 'output/test'

def test_output_writer_save_json():
    """测试JSON保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OutputWriter(output_dir=tmpdir)
        
        test_data = {
            'plate_info': {'type': 'test'},
            'tubes': [{'row': 'A', 'column': 1, 'status': 'filled'}]
        }
        
        writer.save_json(test_data, 'result.json')
        
        # 验证文件存在
        filepath = os.path.join(tmpdir, 'result.json')
        assert os.path.exists(filepath)
        
        # 验证内容
        with open(filepath) as f:
            loaded = json.load(f)
            assert loaded == test_data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_output_writer.py::test_output_writer_init -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal OutputWriter class**

```python
# lib/output_writer.py
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
        with open(filepath, 'w', encoding='utf-8') as f:
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
        
        return {
            'plate_info': plate_info,
            'tubes': tubes_flat,
            'timestamp': timestamp
        }
    
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
        return {
            **stats,
            'verdict': verdict,
            'processing_time_ms': processing_time_ms
        }
    
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
                if tube['centroid']:
                    x, y = tube['centroid']['x'], tube['centroid']['y']
                    color = (0, 255, 0) if tube['status'] == 'filled' else (0, 0, 255)
                    cv2.circle(annotated, (x, y), 8, color, -1)
                    
                    # 标注行列标签
                    label = f"{tube['row']}{tube['column']}"
                    cv2.putText(annotated, label, (x-10, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 绘制统计数据
        filled = sum(1 for row in tube_matrix for tube in row if tube['status'] == 'filled')
        empty = len(tube_matrix) * len(tube_matrix[0]) - filled
        verdict_text = "PASS" if empty == 0 else f"FAIL ({empty} empty)"
        verdict_color = (0, 255, 0) if empty == 0 else (0, 0, 255)
        
        cv2.putText(annotated, f"Filled: {filled}/96", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, verdict_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)
        
        return annotated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_output_writer.py::test_output_writer_init -v`
Expected: PASS

- [ ] **Step 5: Write test for image saving**

```python
# tests/test_output_writer.py (append)
import numpy as np

def test_output_writer_save_image():
    """测试图像保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = OutputWriter(output_dir=tmpdir)
        
        # 创建测试图像
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (255, 0, 0)  # 红色
        
        writer.save_image(test_img, 'test.jpg')
        
        filepath = os.path.join(tmpdir, 'test.jpg')
        assert os.path.exists(filepath)
        
        # 验证图像内容
        loaded = cv2.imread(filepath)
        assert loaded is not None
        assert loaded.shape == test_img.shape
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/hedengfeng/company/cv_handel && python -m pytest tests/test_output_writer.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add lib/output_writer.py tests/test_output_writer.py
git commit -m "feat: add OutputWriter for JSON and image output"
```

---

## Task 6: Main Integration Script

**Files:**
- Create: `detect_liquid_fill.py`

- [ ] **Step 1: Write main integration script**

```python
# detect_liquid_fill.py
"""
液体填充缺陷检测主程序
用于检测96孔微孔板中黑色液体的填充情况
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from lib.droplet_detector import DropletDetector
from lib.grid_detector import GridDetector
from lib.tube_mapper import TubeMapper
from lib.output_writer import OutputWriter


# 配置参数
CONFIG = {
    'threshold_value': 60,
    'min_area': 20,
    'max_area': 2000,
    'standard_count': 96,
    'defect_allowance': 0,
    'roi_top': 0.15,
    'roi_bottom': 0.95,
    'roi_left': 0.12,
    'roi_right': 0.88,
    'gaussian_kernel': 5,
    'rows': 8,
    'cols': 12,
}


def extract_roi(image):
    """
    提取感兴趣区域
    
    Args:
        image: 输入图像
    
    Returns:
        tuple: (roi_image, roi_coords)
    """
    h, w = image.shape[:2]
    
    y_start = int(h * CONFIG['roi_top'])
    y_end = int(h * CONFIG['roi_bottom'])
    x_start = int(w * CONFIG['roi_left'])
    x_end = int(w * CONFIG['roi_right'])
    
    roi = image[y_start:y_end, x_start:x_end]
    
    return roi, (y_start, y_end, x_start, x_end)


def process_image(image_path, output_dir):
    """
    处理单张图像
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
    
    Returns:
        dict: 处理结果
    """
    start_time = datetime.now()
    
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    original = img.copy()
    
    # 2. 提取ROI
    roi, roi_coords = extract_roi(img)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 3. 预处理：高斯模糊
    blurred = cv2.GaussianBlur(roi_gray, (CONFIG['gaussian_kernel'], CONFIG['gaussian_kernel']), 0)
    
    # 4. 液滴检测
    detector = DropletDetector(
        threshold_value=CONFIG['threshold_value'],
        min_area=CONFIG['min_area'],
        max_area=CONFIG['max_area']
    )
    
    contours, thresh, morphed = detector.detect(blurred)
    
    # 5. 网格检测
    grid_detector = GridDetector(rows=CONFIG['rows'], cols=CONFIG['cols'])
    row_bounds, col_bounds = grid_detector.find_grid(morphed)
    
    # 6. 试管映射
    mapper = TubeMapper(rows=CONFIG['rows'], cols=CONFIG['cols'])
    tube_matrix = mapper.map_contours(contours, row_bounds, col_bounds)
    
    # 7. 统计与判定
    stats = mapper.get_statistics(tube_matrix)
    verdict = "PASS" if stats['empty_count'] <= CONFIG['defect_allowance'] else "FAIL"
    
    end_time = datetime.now()
    processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
    
    # 8. 输出结果
    writer = OutputWriter(output_dir)
    
    # 从文件名提取plate信息
    basename = os.path.basename(image_path)
    plate_type = basename.split('-')[0] + '-' + basename.split('-')[1] if '-' in basename else 'unknown'
    
    result_json = writer.generate_result_json(
        tube_matrix,
        {'type': plate_type, 'rows': CONFIG['rows'], 'columns': CONFIG['cols'], 'total_tubes': CONFIG['standard_count']},
        start_time.isoformat()
    )
    
    stats_json = writer.generate_stats_json(stats, verdict, processing_time_ms)
    
    # 保存JSON
    writer.save_json(result_json, 'result.json')
    writer.save_json(stats_json, 'stats.json')
    
    # 保存中间图像
    writer.save_image(roi, '01_roi.jpg')
    writer.save_image(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), '02_preprocessed.jpg')
    writer.save_image(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), '03_threshold.jpg')
    writer.save_image(cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR), '04_morphology.jpg')
    
    # 绘制轮廓图
    contours_img = roi.copy()
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 1)
    writer.save_image(contours_img, '05_contours.jpg')
    
    # 绘制网格图
    grid_img = roi.copy()
    for y in row_bounds:
        cv2.line(grid_img, (0, y), (grid_img.shape[1], y), (100, 100, 100), 1)
    for x in col_bounds:
        cv2.line(grid_img, (x, 0), (x, grid_img.shape[0]), (100, 100, 100), 1)
    writer.save_image(grid_img, '06_grid.jpg')
    
    # 绘制映射图
    mapping_img = writer.draw_annotated_image(roi, tube_matrix, row_bounds, col_bounds)
    writer.save_image(mapping_img, '07_mapping.jpg')
    
    # 最终标注图（包含统计数据）
    final_img = writer.draw_annotated_image(roi, tube_matrix, row_bounds, col_bounds)
    writer.save_image(final_img, '08_annotated.jpg')
    
    return {
        'stats': stats,
        'verdict': verdict,
        'output_dir': output_dir
    }


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python detect_liquid_fill.py <图像路径>")
        print("示例: python detect_liquid_fill.py sample/96-50-黑色/96-50-2.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 验证文件存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)
    
    # 创建输出目录
    basename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = f"output/{basename}-{timestamp}"
    
    # 处理图像
    print(f"处理图像: {image_path}")
    result = process_image(image_path, output_dir)
    
    # 输出结果
    print(f"\n检测结果:")
    print(f"  总tube数: {result['stats']['total_tubes']}")
    print(f"  填充数: {result['stats']['filled_count']}")
    print(f"  空tube数: {result['stats']['empty_count']}")
    print(f"  填充率: {result['stats']['fill_rate']:.2f}%")
    print(f"  判定结果: {result['verdict']}")
    print(f"\n输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on empty plate**

Run: `cd /Users/hedengfeng/company/cv_handel && python detect_liquid_fill.py sample/96-50-黑色/96-50-1.jpg`
Expected: ~0 filled, ~96 empty, FAIL

- [ ] **Step 3: Test on full plate**

Run: `cd /Users/hedengfeng/company/cv_handel && python detect_liquid_fill.py sample/96-50-黑色/96-50-2.jpg`
Expected: ≥90 filled, ≤6 empty, PASS or FAIL depending on detection quality

- [ ] **Step 4: Test on reference plate with defects**

Run: `cd /Users/hedengfeng/company/cv_handel && python detect_liquid_fill.py sample/96-50-黑色/96-50-参照.jpg`
Expected: ~88 filled, 8 empty, FAIL

- [ ] **Step 5: Verify output structure**

Run: `ls -la output/*/`
Expected: Each output directory contains result.json, stats.json, and 8 intermediate images (01-08_*.jpg)

- [ ] **Step 6: Commit**

```bash
git add detect_liquid_fill.py
git commit -m "feat: add main detection script with full pipeline"
```

---

## Task 7: Verification and Documentation

**Files:**
- None (verification only)

- [ ] **Step 1: Verify JSON output format**

Run: `cat output/*/result.json | head -50`
Expected: Valid JSON with plate_info, tubes array (96 entries), timestamp

- [ ] **Step 2: Verify stats output**

Run: `cat output/*/stats.json`
Expected: total_tubes, filled_count, empty_count, fill_rate, verdict, processing_time_ms

- [ ] **Step 3: Check intermediate images quality**

Manually review: Open 01-08_*.jpg images in output directory
Expected: 
- 01_roi.jpg: Properly cropped (no orange strip, no screws)
- 03_threshold.jpg: Black regions shown as white
- 05_contours.jpg: Valid contours outlined
- 08_annotated.jpg: Green circles (filled), red circles (empty), stats text visible

- [ ] **Step 4: Validate against reference defects**

Compare: `96-50-参照.jpg` red circle positions with `08_annotated.jpg` red circles
Expected: 8 empty tubes detected near the marked positions

- [ ] **Step 5: Final commit with all changes**

```bash
git add -A
git commit -m "feat: complete liquid fill detection system

- DropletDetector: threshold segmentation for black liquid
- GridDetector: projection analysis for 8x12 grid
- TubeMapper: contour-to-position mapping
- OutputWriter: JSON reports + 8 intermediate images
- Main script: full pipeline integration

Verified on sample images:
- Empty plate: ~0 detected
- Full plate: ≥90 detected
- Reference (8 defects): ~88 detected"
```

---

## Self-Review Checklist

**Spec Coverage:**
- [x] Architecture: 4 components ✓ (Tasks 2-5)
- [x] Data Flow: 8-stage pipeline ✓ (Task 6)
- [x] Intermediate images: 8 outputs ✓ (Task 5, 6)
- [x] Error handling: File load, ROI, grid fallback ✓ (Task 6)
- [x] Testing: 3 test images ✓ (Task 6 steps 2-4)
- [x] Output format: JSON structure ✓ (Task 5)

**Placeholder Scan:**
- [x] No TBD, TODO, or vague requirements
- [x] All code blocks contain complete implementation
- [x] All test cases have actual assertions
- [x] All commands specify expected output

**Type Consistency:**
- [x] tube_matrix structure consistent across TubeMapper and OutputWriter
- [x] centroid dict keys ('x', 'y') used consistently
- [x] CONFIG parameters match across all modules

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-10-liquid-fill-detection.md`**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?