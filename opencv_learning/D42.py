import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning

翻轉：實作上下翻轉
縮放：實作鄰近差值
平移：建立 Translation Transformation Matrix 來做平移
"""

# 讀取 csv 文件
df = pd.read_csv("data/facial-keypoints-detection/training.csv")

# 前 5 筆資料, .T 的作用是轉置，如果不理解可以和 data.head() 的結果相比較
# print(df.head().T)

# 過濾有缺失值的 row
df = df.dropna()

# 將圖片像素值讀取為 numpy array 的形態
df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values

# 單獨把圖像 array 抽取出來
imgs = np.vstack(df['Image'].values) / 255

# reshape 為 96 x 96
imgs = imgs.reshape(df.shape[0], 96, 96)

# 轉換為 float
imgs = imgs.astype(np.float32)

# 提取坐標的部分
points = df[df.columns[:-1]].values

# 轉換為 float
points = points.astype(np.float32)

# normalize 坐標值
points = points / 96

print("圖像資料:", imgs.shape)
print("關鍵點資料:", points.shape)
# 圖像資料: (2140, 96, 96)
# 關鍵點資料: (2140, 30)

sample_img = imgs[0]
sample_points = points[0]

n_image, height, width = imgs.shape

points *= width
points = np.int(width)

n_coord = int(len(sample_points) / 2)

for i in range(n_coord):
    cv2.circle(sample_img, (sample_points[i], sample_points[i + 1]), 3, color=(0, 0, 255))

sample_img = cv2.resize(sample_img, (400, 400), interpolation=cv2.INTER_CUBIC)
showImages(sample_img=sample_img)
# plt.scatter(x_points, y_points)
# plt.imshow(sample_img, cmap='gray')
