import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Working directory: CupoyLearning"""


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    boxes = np.array(bounding_boxes)

    '''取出每一個 BOX 的 x1, y1, x2, y2'''
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    '''預設 list 用來保存 BBOX '''
    picked_boxes = []
    picked_score = []

    '''計算每一個 BOX 的面積'''
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    '''排列 BOX 分數'''
    order = np.argsort(score)
    print(order)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        '''當前 confidence 最大的預測框'''
        index = order[-1]

        # Pick the bounding box with largest confidence score
        '''保存這個 BBOX'''
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        '''計算 Boxes 與最高分 BOX 之間的 IOU'''
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        print("intersection:", intersection)

        '''計算 IOU '''
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        print("ratio:", ratio)

        '''重疊率小於預測 threshold 的 Boxes 要保存下來'''
        left = np.where(ratio < threshold)
        order = order[left]
        print(order)

    return picked_boxes, picked_score


path = 'image/nms.jpg'

# Bounding boxes
bounding_boxes = [(200, 82, 350, 317), (180, 67, 305, 282), (200, 90, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]

# Read image
image = cv2.imread(path)

# Copy image as original
org = image.copy()

# Draw parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

'''IoU threshold :重疊超過多少就過濾掉'''
threshold = 0.5

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
picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

# Draw bounding boxes and confidence score after non-maximum supression
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

plt.imshow(image[:, :, ::-1])
plt.show()
