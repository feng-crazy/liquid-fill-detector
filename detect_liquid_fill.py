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


CONFIG = {
    "threshold_value": 55,
    "min_area": 10,
    "max_area": 5000,
    "standard_count": 96,
    "defect_allowance": 0,
    "roi_top": 0.15,
    "roi_bottom": 0.95,
    "roi_left": 0.12,
    "roi_right": 0.88,
    "gaussian_kernel": 5,
    "rows": 8,
    "cols": 12,
    "use_adaptive": False,
    "adaptive_block_size": 35,
    "adaptive_c": 2,
    "use_opening_preprocess": False,
    "opening_kernel_size": 3,
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

    y_start = int(h * CONFIG["roi_top"])
    y_end = int(h * CONFIG["roi_bottom"])
    x_start = int(w * CONFIG["roi_left"])
    x_end = int(w * CONFIG["roi_right"])

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
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    original = img.copy()
    roi, roi_coords = extract_roi(img)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        roi_gray, (CONFIG["gaussian_kernel"], CONFIG["gaussian_kernel"]), 0
    )

    detector = DropletDetector(
        threshold_value=CONFIG["threshold_value"],
        min_area=CONFIG["min_area"],
        max_area=CONFIG["max_area"],
        use_adaptive=CONFIG["use_adaptive"],
        adaptive_block_size=CONFIG["adaptive_block_size"],
        adaptive_c=CONFIG["adaptive_c"],
        use_opening_preprocess=CONFIG["use_opening_preprocess"],
        opening_kernel_size=CONFIG["opening_kernel_size"],
    )

    contours, thresh, morphed = detector.detect(blurred)

    grid_detector = GridDetector(rows=CONFIG["rows"], cols=CONFIG["cols"])
    row_bounds, col_bounds = grid_detector.find_grid(morphed)

    mapper = TubeMapper(rows=CONFIG["rows"], cols=CONFIG["cols"])
    tube_matrix = mapper.map_contours(contours, row_bounds, col_bounds)

    stats = mapper.get_statistics(tube_matrix)
    verdict = "PASS" if stats["empty_count"] <= CONFIG["defect_allowance"] else "FAIL"

    end_time = datetime.now()
    processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

    writer = OutputWriter(output_dir)

    basename = os.path.basename(image_path)
    plate_type = (
        basename.split("-")[0] + "-" + basename.split("-")[1]
        if "-" in basename
        else "unknown"
    )

    result_json = writer.generate_result_json(
        tube_matrix,
        {
            "type": plate_type,
            "rows": CONFIG["rows"],
            "columns": CONFIG["cols"],
            "total_tubes": CONFIG["standard_count"],
        },
        start_time.isoformat(),
    )

    stats_json = writer.generate_stats_json(stats, verdict, processing_time_ms)

    writer.save_json(result_json, "result.json")
    writer.save_json(stats_json, "stats.json")

    writer.save_image(roi, "01_roi.jpg")
    writer.save_image(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), "02_preprocessed.jpg")
    writer.save_image(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "03_threshold.jpg")
    writer.save_image(cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR), "04_morphology.jpg")

    contours_img = roi.copy()
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 1)
    writer.save_image(contours_img, "05_contours.jpg")

    grid_img = roi.copy()
    for y in row_bounds:
        cv2.line(grid_img, (0, y), (grid_img.shape[1], y), (100, 100, 100), 1)
    for x in col_bounds:
        cv2.line(grid_img, (x, 0), (x, grid_img.shape[0]), (100, 100, 100), 1)
    writer.save_image(grid_img, "06_grid.jpg")

    mapping_img = writer.draw_annotated_image(roi, tube_matrix, row_bounds, col_bounds)
    writer.save_image(mapping_img, "07_mapping.jpg")

    final_img = writer.draw_annotated_image(roi, tube_matrix, row_bounds, col_bounds)
    writer.save_image(final_img, "08_annotated.jpg")

    return {"stats": stats, "verdict": verdict, "output_dir": output_dir}


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python detect_liquid_fill.py <图像路径>")
        print("示例: python detect_liquid_fill.py sample/96-50-黑色/96-50-2.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    basename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/{basename}-{timestamp}"

    print(f"处理图像: {image_path}")
    result = process_image(image_path, output_dir)

    print(f"\n检测结果:")
    print(f"  总tube数: {result['stats']['total_tubes']}")
    print(f"  填充数: {result['stats']['filled_count']}")
    print(f"  空tube数: {result['stats']['empty_count']}")
    print(f"  填充率: {result['stats']['fill_rate']:.2f}%")
    print(f"  判定结果: {result['verdict']}")
    print(f"\n输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
