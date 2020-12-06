import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning

Hint: 人物原始邊框座標 (60, 40), (420, 510)

請根據 Lena 圖做以下處理

* 對明亮度做直方圖均衡處理
* 水平鏡像 + 縮放處理 (0.5 倍)
* 畫出人物矩形邊框
"""

path = "image/lena.png"
img = cv2.imread(path)

"""
Hint: 矩形
"""

img_rect = img.copy()
cv2.rectangle(img_rect, (60, 40), (420, 510), (0, 0, 255), 3)

showImages(img=img, img_rect=img_rect)

"""
Hint: 線
"""

img_line = img.copy()
cv2.line(img_line, (60, 40), (420, 510), (0, 0, 255), 3)

showImages(img=img, img_line=img_line)

"""
Hint: 文字
"""

img_text = img.copy()
cv2.putText(img_text, '(60, 40)', (60, 40), 0, 1, (0, 0, 255), 2)

showImages(img=img, img_text=img_text)
