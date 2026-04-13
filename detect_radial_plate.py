"""
径向微孔板液体填充检测主程序
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from lib.radial_detector import RadialDetector
from lib.output_writer import OutputWriter


CONFIG = {
    "roi_top": 0.15,
    "roi_bottom": 0.95,
    "roi_left": 0.12,
    "roi_right": 0.88,
    "rows": 8,
    "cols": 12,
    "brightness_threshold": 100,
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

    detector = RadialDetector(rows=CONFIG["rows"], cols=CONFIG["cols"])
    result = detector.detect_plate(roi)

    stats = result["stats"]
    verdict = "PASS" if stats["empty_count"] <= 8 else "FAIL"

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
        result["tube_matrix"],
        {
            "type": plate_type,
            "rows": CONFIG["rows"],
            "columns": CONFIG["cols"],
            "total_tubes": CONFIG["rows"] * CONFIG["cols"],
        },
        start_time.isoformat(),
    )

    stats_json = writer.generate_stats_json(stats, verdict, processing_time_ms)

    writer.save_json(result_json, "result.json")
    writer.save_json(stats_json, "stats.json")

    writer.save_image(roi, "01_roi.jpg")

    circles_img = roi.copy()
    for contour_data in result["contours"]:
        cx, cy, area, cnt = contour_data
        cv2.drawContours(circles_img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(circles_img, (cx, cy), 3, (0, 0, 255), -1)
    writer.save_image(circles_img, "02_contours.jpg")

    annotated_img = roi.copy()
    for row in result["tube_matrix"]:
        for tube in row:
            if tube["centroid"]:
                x, y = tube["centroid"]["x"], tube["centroid"]["y"]
                color = (0, 255, 0) if tube["status"] == "filled" else (255, 0, 0)
                cv2.circle(annotated_img, (x, y), 5, color, -1)
    writer.save_image(annotated_img, "03_annotated.jpg")

    return {"stats": stats, "verdict": verdict, "output_dir": output_dir}


def main():
    if len(sys.argv) < 2:
        print("用法: python detect_radial_plate.py <图像路径>")
        print("示例: python detect_radial_plate.py sample/96-50-黑色/96-50-2.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    basename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/{basename}-radial-{timestamp}"

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
