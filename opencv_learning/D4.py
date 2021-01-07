import time

import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning

翻轉：實作上下翻轉
縮放：實作鄰近差值
平移：建立 Translation Transformation Matrix 來做平移
"""

path = "data/image/lena.png"
img = cv2.imread(path)

"""
上下翻轉圖片
"""

# 水平翻轉 (horizontal)
horizontal_flip = img[:, ::-1, :]

# 垂直翻轉 (vertical)
vertical_flip = img[::-1, :, :]

showImages(img=img, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)

"""
縮放圖片

放大
我們先透過縮小圖片去壓縮原有圖片保有的資訊，再放大比較不同方法之間的速度與圖片品質
"""

# 將圖片縮小成原本的 20%
small = cv2.resize(img.copy(), None, fx=0.2, fy=0.2)

# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大
fx, fy = 8, 8

# 鄰近差值 scale + 計算花費時間
start_time = time.time()
img_area_scale = cv2.resize(small, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
print('INTER_NEAREST zoom cost {}'.format(time.time() - start_time))

# 雙立方差補 scale + 計算花費時間
start_time = time.time()
img_cubic_scale = cv2.resize(small, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
print('INTER_CUBIC zoom cost {}'.format(time.time() - start_time))

origin_img = cv2.resize(img, img_area_scale.shape[:2])

showImages(origin_img=origin_img, img_area_scale=img_area_scale, img_cubic_scale=img_cubic_scale)

"""
平移幾何轉換
"""

# 設定 translation transformation matrix
# x 平移 100 pixel; y 平移 50 pixel
M = np.array([[1, 0, 100],
              [0, 1, 50]], dtype=np.float32)
shift_img1 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# x 平移 50 pixel; y 平移 100 pixel
M = np.array([[1, 0, 50],
              [0, 1, 100]], dtype=np.float32)
shift_img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

showImages(img=img, shift_img1=shift_img1, shift_img2=shift_img2)
