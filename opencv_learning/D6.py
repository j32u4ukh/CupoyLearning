import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning
仿射變換：保證「共線不變性」與「比例不變性」
"""

path = "image/lena.png"
img = cv2.imread(path)

"""
Affine Transformation - Case 1: rotation 45 -> scale 0.5 -> shift (x+100, y-50)
"""

height, width, _ = img.shape

# 取得旋轉矩陣
# getRotationMatrix2D(center, angle, scale)
M_rotate = cv2.getRotationMatrix2D((width // 2, height // 2), 45, 0.5)
print('Rotation Matrix')
print(M_rotate)

# 取得平移矩陣
M_translate = np.array([[1, 0, 100],
                        [0, 1, -50]], dtype=np.float32)
print('Translation Matrix')
print(M_translate)

# 旋轉
img_rotate = cv2.warpAffine(img, M_rotate, (height, width))

# 平移
img_rotate_trans = cv2.warpAffine(img_rotate, M_translate, (height, width))

showImages(img=img, img_rotate=img_rotate, img_rotate_trans=img_rotate_trans)

"""
Affine Transformation - Case 2: any three point
"""

# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
height, width, _ = img.shape

points = np.array([[50, 50], [300, 100], [200, 300]], dtype=np.float32)
points_prime = np.array([[80, 80], [330, 150], [300, 300]], dtype=np.float32)

# 取得 affine 矩陣並做 affine 操作
M_affine = cv2.getAffineTransform(points, points_prime)
img_affine = cv2.warpAffine(img, M_affine, (height, width))

showImages(img=img, img_affine=img_affine)
