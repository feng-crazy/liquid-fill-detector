针对黑色液体场景，使用传统机器视觉，因为黑色液滴与白色/金属背景有极高的对比度。

我们需要精确找到每一个孔的位置，统计**“黑色液滴”的数量**。如果液滴数量明显少于总数（例如96个），就说明有试管没加液体。

核心思路
ROI提取：排除顶部橙色盖子和边缘螺丝的干扰，只关注中间排列针头的区域。
阈值分割：提取图像中的黑色区域（液滴）。
形态学处理：去除噪点，确保液滴是完整的连通域。
轮廓计数：统计符合大小特征的黑色轮廓数量。
判定：如果数量 < 阈值，则判定整板有缺陷。
Python + OpenCV 代码实现

参考代码
```python

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_black_droplets(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 定义 ROI (感兴趣区域)
    # 观察图片，针头主要分布在中间。我们需要排除顶部的橙色部分和边缘的螺丝。
    # 这里通过简单的坐标裁剪来模拟，实际项目中可以通过边缘检测自动获取
    h, w = gray.shape
    # 根据图片比例估算 ROI：去掉顶部约15%，左右各去掉10%，底部去掉5%
    # 这是一个粗略的估计，针对你提供的图片进行调整
    y_start = int(h * 0.15)
    y_end = int(h * 0.95)
    x_start = int(w * 0.12)
    x_end = int(w * 0.88)
    
    roi_gray = gray[y_start:y_end, x_start:x_end]
    roi_color = original_img[y_start:y_end, x_start:x_end]
    
    # 3. 预处理与阈值分割 (针对黑色液体)
    # 黑色液体在灰度图中值很低。我们使用阈值提取黑色物体。
    # 阈值设为 60，低于60的被认为是黑色液滴 (根据图片2观察，液滴非常黑)
    _, thresh = cv2.threshold(roi_gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # 4. 形态学操作
    # 使用开运算去除小的噪点
    kernel = np.ones((3,3),np.uint8)
    morphed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 使用闭运算连接可能断开的液滴
    morphed_thresh = cv2.morphologyEx(morphed_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 查找轮廓并计数
    contours, _ = cv2.findContours(morphed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_droplets = []
    min_area = 20  # 最小面积过滤噪点
    max_area = 2000 # 最大面积过滤大块干扰
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # 计算轮廓中心 (相对于ROI的坐标)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                valid_droplets.append((cX, cY, area))
    
    # 6. 结果统计与判定
    total_count = len(valid_droplets)
    # 假设标准板子有 96 个孔 (8行x12列)
    STANDARD_COUNT = 96 
    ALLOWANCE = 5 
    
    is_defect = total_count < (STANDARD_COUNT - ALLOWANCE)
    
    # 7. 可视化
    # 在 ROI 图上画出检测到的液滴
    for i, (cx, cy, area) in enumerate(valid_droplets):
        # 画绿色的圈表示检测到液滴
        cv2.circle(roi_color, (cx, cy), 8, (0, 255, 0), -1)
    
    # 拼接 ROI 和 阈值图用于展示
    display_thresh = cv2.cvtColor(morphed_thresh, cv2.COLOR_GRAY2BGR)
    combined_view = np.hstack((roi_color, display_thresh))
    
    # 添加文字信息
    status_text = "PASS" if not is_defect else "FAIL (Missing Droplets)"
    color = (0, 255, 0) if not is_defect else (0, 0, 255)
    cv2.putText(roi_color, f"Count: {total_count}/{STANDARD_COUNT}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(roi_color, status_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # 显示结果
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image with ROI marked")
    plt.axis('off')
    
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Droplets (Green dots) - Status: {status_text}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"检测完成。")
    print(f"检测到液滴数量: {total_count}")
    print(f"判定结果: {'合格' if not is_defect else '不合格 (存在未填充试管)'}")

# 使用示例
# 请替换为你的图片路径
# 这里假设你保存了图片2为 'image_full_black.jpg'
# detect_black_droplets('image_full_black.jpg') 

# 为了演示，我假设你直接把图片传进来运行。
# 如果你是在本地运行，请取消下面注释并填入路径
# detect_black_droplets('path_to_your_image.jpg')

```

代码逻辑解释：
ROI裁剪：

代码中 y_start, x_start 等参数是为了把针头区域单独切出来。
目的：图片上方有两个黑色的螺丝，如果不切掉，会被误判为液滴。切掉后只保留中间密集的针头区。
阈值分割 (cv2.threshold)：

cv2.THRESH_BINARY_INV：因为我们要找黑色的东西。
参数 60：灰度值小于60的像素变成白色（前景），大于60的变成黑色（背景）。这能有效提取黑色液滴。
轮廓过滤：

通过 area (面积) 过滤。
太小的（噪点）不要，太大的（可能是阴影或大块污渍）不要。只保留针头尖端大小的黑点。
判定逻辑：

如果你知道板子总共有多少个孔（比如96孔板），你可以设置 STANDARD_COUNT = 96。
如果检测到的黑点数量远小于96（比如只有80个），说明有16个孔没加液体，判定为 FAIL。
针对你图片的测试预期：
图片1（空载）：检测到的黑色液滴数量应该接近 0。 -> FAIL (或者根据你的逻辑，如果是做空载检测，那就是PASS，看你的具体需求，通常是要检测“是否充满了”，所以空载就是没充满)。
图片2（全黑液）：检测到的数量应该接近 96（假设是96孔板）。 -> PASS。
图片3（有缺陷）：检测到的数量应该是 96 减去 红圈的数量。如果红圈有8个，那检测到88个。 -> FAIL。
下一步建议：
你可以先用这张代码跑一下图片2和图片3。

如果图片2能数出大概90-100个点，说明阈值和ROI设置正确。
如果图片3数出来的点明显少于图片2，说明方案有效。
如果效果不好（比如把针头的杆部也识别进去了），我们需要调整阈值或者ROI的大小。