import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Working directory: CupoyLearning

NMS 在 YOLO 的實際運作以每一個類別為主，各別執行NMS。
YOLO 在NMS中採用的信心度為「每個 bbox 包含各類別的信心度」

作業
在 NMS 流程中，IoU 重疊率參數(nms_threshold)調高，試著思考一下輸出的預測框會有甚麼變化?
Hint: 可以回頭看今天介紹的內容，思考輸出的預測框變多或變少?

Ans: 門檻調高，可被保留的預測框數量則會增加
"""


def softNms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    """預設 list 用來保存 BBOX """
    picked_boxes = []
    picked_score = []

    # 所有候選框
    boxes = np.array(bounding_boxes)

    '''取出每一個 BOX 的 x1, y1, x2, y2'''
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    '''計算每一個 BOX 的面積'''
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    '''排列 BOX 分數'''
    order = np.argsort(score)
    print(order)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        '''當前 confidence 最大的預測框
        第一輪是全體最大，第二輪則是排除最大以及與最大之重疊率過高的其他框後的最大，以此類推。
        '''
        index = order[-1]
        others = order[:-1]

        # Pick the bounding box with largest confidence score
        '''保存這個 BBOX'''
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])

        # Compute ordinates of intersection-over-union(IOU)
        '''計算 Boxes 與最高分 BOX 之間的 IOU'''
        x1 = np.maximum(start_x[index], start_x[others])
        x2 = np.minimum(end_x[index], end_x[others])
        y1 = np.maximum(start_y[index], start_y[others])
        y2 = np.minimum(end_y[index], end_y[others])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        print("intersection:", intersection)

        ''' 計算和當前 confidence 最高的框的 IOU '''
        iou_ratio = intersection / (areas[index] + areas[others] - intersection)
        print("iou_ratio:", iou_ratio)

        '''
        重疊率過高，表示兩個框'可能'圈到同一物體，Soft-NMS 會將該框的分數調降，而非排除
        
        B = {b1, b2, ..., bn} 所有候選框(box)
        S = confidence score
        M = 當前循環 S 最大的框，將被放入 D 當中
        D = [] 用於存放選選出來的 box
        bi = 第 i 個候選框(box)
        Si = 第 i 個 confidence score
        Si = Si * e^(-(iou(M, bi)^2 / std))
        '''
        std = np.std(iou_ratio) + 1e-8
        score = score[others]
        boxes = boxes[others]
        modify_index = np.where(iou_ratio > threshold)
        modify = np.ones_like(score)
        modify[modify_index] = np.exp(-np.power(iou_ratio, 2.0) / std)[modify_index]
        score *= modify

        # 排除當前循環 confidence score 最大的框
        order = np.argsort(score)
        print(order)

    return picked_boxes, picked_score


path = 'data/image/nms.jpg'

# Bounding boxes
bounding_boxes = [(200, 82, 350, 317), (180, 67, 305, 282), (200, 90, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]

# IoU threshold
threshold = 0.5

# Read image
image = cv2.imread(path)

# Copy image as original
org = image.copy()

# Draw parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

"""經過 NMS 之前"""

# Draw bounding boxes and confidence score
for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

# Show image
plt.imshow(org[:, :, ::-1])
plt.show()

"""經過 NMS 之後"""

# Run non-max suppression algorithm
picked_boxes, picked_score = softNms(bounding_boxes, confidence_score, threshold)
print("#picked_boxes:", len(picked_boxes))
# #picked_boxes: 3

# Draw bounding boxes and confidence score after non-maximum supression
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

plt.imshow(image[:, :, ::-1])
plt.show()
