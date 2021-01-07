import cv2
import matplotlib.patches as patches

import matplotlib.pyplot as plt

"""Working directory: CupoyLearning"""


def rectToCrood(rect):
    """
    X, Y, W, H -> X0, Y0, X1, Y1

    :param rect:
    :return:
    """
    return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]


""" 定義 IOU 計算 """


def computeIOU(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    x_right = min(box1[2], box2[2])

    y_below = max(box1[1], box2[1])
    y_top = min(box1[3], box2[3])

    # 計算交集區域
    intersection = max(0, x_right - x_left + 1) * max(0, y_top - y_below + 1)

    # 計算各自的 BBOX 大小
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 計算IOU
    iou = intersection / (area1 + area2 - intersection)

    # return the intersection over union value
    return iou


path = "image/Dog.JPG"

# 讀入照片
image = cv2.imread(path)

# 因為CV2會將照片讀成BGR，要轉回來
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
先設 BBOX 格式為 [X, Y, W, H]

也就是左上角那一點的座標以及 BBOX 的寬和高
"""

ground_truth_bbox = [1900, 700, 1800, 1800]
prediction_bbox = [1800, 800, 1500, 1500]

""" 轉換成 [X0, Y0, X1, Y1] """

ground_truth_crood = rectToCrood(ground_truth_bbox)
prediction_crood = rectToCrood(prediction_bbox)

plt.rcParams['figure.figsize'] = (20, 10)
fig, ax = plt.subplots(1)

# 畫出圖片
ax.imshow(image)

# 畫 BBOX-Prediction
box1 = patches.Rectangle((prediction_bbox[0], prediction_bbox[1]), prediction_bbox[2], prediction_bbox[3],
                         linewidth=3, edgecolor='r', facecolor='none', )
ax.text(1800, 800, 'Prediction', withdash=True, size=20)

# 畫 BBOX-Ground_Truth
box2 = patches.Rectangle((ground_truth_bbox[0], ground_truth_bbox[1]), ground_truth_bbox[2], ground_truth_bbox[3],
                         linewidth=3, edgecolor='b', facecolor='none')
ax.text(1900, 700, 'Ground Truth', withdash=True, size=20)

# Add the patch to the Axes
ax.add_patch(box1)
ax.add_patch(box2)

plt.show()

iou = computeIOU(box1=ground_truth_crood, box2=prediction_crood)

# IOU值： 0.6196482836879266
print('IOU值：', iou)
