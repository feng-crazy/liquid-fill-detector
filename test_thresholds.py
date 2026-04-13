"""
测试不同阈值对检测结果的影响
"""

import sys
import os

import detect_liquid_fill

test_thresholds = [40, 50, 60, 70, 80]
test_images = [
    "sample/96-50-黑色/96-50-1.jpg",
    "sample/96-50-黑色/96-50-2.jpg",
    "sample/96-50-黑色/96-50-参照.jpg",
]

expected = {
    "96-50-1.jpg": {"min_filled": 0, "max_filled": 5, "desc": "Empty plate"},
    "96-50-2.jpg": {"min_filled": 90, "max_filled": 96, "desc": "Full plate"},
    "96-50-参照.jpg": {
        "min_filled": 80,
        "max_filled": 96,
        "desc": "Reference plate (8 defects)",
    },
}

print("=" * 80)
print("阈值测试 - 验证参数调整是否能改善检测")
print("=" * 80)

for threshold in test_thresholds:
    print(f"\n{'=' * 80}")
    print(f"阈值: {threshold}")
    print(f"{'=' * 80}")

    detect_liquid_fill.CONFIG["threshold_value"] = threshold

    for image_path in test_images:
        basename = os.path.basename(image_path)
        exp = expected[basename]

        output_dir = f"output/test-{threshold}-{basename.replace('.jpg', '')}"
        result = detect_liquid_fill.process_image(image_path, output_dir)

        filled = result["stats"]["filled_count"]
        in_range = exp["min_filled"] <= filled <= exp["max_filled"]

        status = "✓" if in_range else "✗"
        print(
            f"{status} {exp['desc']}: {filled} filled (expected {exp['min_filled']}-{exp['max_filled']})"
        )

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("如果所有阈值都无法使检测结果接近预期值，")
print("则证明问题根源是架构设计（矩形网格 vs 径向布局），")
print("而非参数设置问题。")
