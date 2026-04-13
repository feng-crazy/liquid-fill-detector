"""
填充率分析检测主程序
结合阈值分割和网格填充率分析进行缺陷检测
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from lib.droplet_detector import DropletDetector
from lib.grid_detector import GridDetector
from lib.fill_rate_analyzer import FillRateAnalyzer
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
    "k_factor": 3.0,
    "min_fill_threshold": 0.01,
}


def extract_roi(image):
    h, w = image.shape[:2]
    y_start = int(h * CONFIG["roi_top"])
    y_end = int(h * CONFIG["roi_bottom"])
    x_start = int(w * CONFIG["roi_left"])
    x_end = int(w * CONFIG["roi_right"])
    roi = image[y_start:y_end, x_start:x_end]
    return roi, (y_start, y_end, x_start, x_end)


def process_image(image_path, output_dir):
    start_time = datetime.now()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

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

    analyzer = FillRateAnalyzer(
        rows=CONFIG["rows"],
        cols=CONFIG["cols"],
        k_factor=CONFIG["k_factor"],
        min_fill_threshold=CONFIG["min_fill_threshold"],
    )

    result = analyzer.analyze(morphed, row_bounds, col_bounds)

    stats = result["stats"]
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

    tube_matrix = []
    cell_details = analyzer.get_cell_details(
        result["fill_rates"], result["filled_matrix"]
    )
    for detail in cell_details:
        tube_matrix.append(
            {
                "row": detail["row"],
                "column": detail["column"],
                "status": detail["status"],
                "centroid": None,
                "droplet_area": None,
                "fill_rate": detail["fill_rate"],
            }
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

    grid_img = roi.copy()
    for y in row_bounds:
        cv2.line(grid_img, (0, y), (grid_img.shape[1], y), (100, 100, 100), 1)
    for x in col_bounds:
        cv2.line(grid_img, (x, 0), (x, grid_img.shape[0]), (100, 100, 100), 1)
    writer.save_image(grid_img, "05_grid.jpg")

    annotated_img = roi.copy()
    for detail in cell_details:
        r = ord(detail["row"]) - ord("A")
        c = detail["column"] - 1
        cx = col_bounds[c]
        cy = row_bounds[r]
        color = (0, 255, 0) if detail["status"] == "filled" else (255, 0, 0)
        cv2.circle(annotated_img, (cx, cy), 5, color, -1)
        fill_rate_text = f"{detail['fill_rate']:.2f}"
        cv2.putText(
            annotated_img,
            fill_rate_text,
            (cx - 20, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )
    writer.save_image(annotated_img, "06_annotated.jpg")

    return {
        "stats": stats,
        "verdict": verdict,
        "output_dir": output_dir,
        "fill_rates": result["fill_rates"],
        "dynamic_threshold": result["dynamic_threshold"],
        "mean_fill_rate": result["mean_fill_rate"],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python detect_fill_rate.py <图像路径>")
        print("示例: python detect_fill_rate.py sample/96-50-黑色/96-50-2.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    basename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/{basename}-fillrate-{timestamp}"

    print(f"处理图像: {image_path}")
    result = process_image(image_path, output_dir)

    print(f"\n检测结果:")
    print(f"  总tube数: {result['stats']['total_tubes']}")
    print(f"  填充数: {result['stats']['filled_count']}")
    print(f"  空tube数: {result['stats']['empty_count']}")
    print(f"  填充率: {result['stats']['fill_rate']:.2f}%")
    print(f"  平均填充率: {result['mean_fill_rate']:.4f}")
    print(f"  动态阈值: {result['dynamic_threshold']:.4f}")
    print(f"  判定结果: {result['verdict']}")
    print(f"\n输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
