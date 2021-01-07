import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning

根據以下的參考點，嘗試做透視變換

point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)
"""

path = "data/image/lena.png"
img = cv2.imread(path)

"""
透視轉換
"""

height, width, _ = img.shape

# 設定四對點，並取得 perspective 矩陣
w = 120
point1 = np.array([[60, 40], [420, 40], [420, 510], [60, 510]], dtype=np.float32)
point2 = np.array([[0, 80], [w, 120], [w, 430], [0, 470]], dtype=np.float32)

# 計算 透視變換 Perspective Transformation 之矩陣
M = cv2.getPerspectiveTransform(point1, point2)
print("M\n", M)

# perspective 轉換
img_perspective = cv2.warpPerspective(img, M, (width, height))

showImages(img=img, img_perspective=img_perspective)
