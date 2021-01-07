import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Dense

from utils.dl import VGG16

"""Working directory: CupoyLearning

請嘗試使用 keras 來定義一個直接預測 15 個人臉關鍵點坐標的檢測網路，以及適合這個網路的 loss function
"""


def loadData(file_name):
    # 讀取 csv 文件
    df = pd.read_csv(f"data/facial-keypoints-detection/{file_name}.csv")

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
    points = points / 96 - 0.5

    return imgs, points


imgs, points = loadData(file_name="training")
print("圖像資料:", imgs.shape)
print("關鍵點資料:", points.shape)

# 定義人臉關鍵點檢測網路
vgg16 = VGG16(include_top=False, input_shape=(96, 96, 1), pooling='avg')
# cnn_layer = Model(inputs=vgg16.layers[0].input, outputs=vgg16.layers[10].output)
x = vgg16.output
# x = Dense(100, activation="relu")(x)
predictions = Dense(30)(x)
model = Model(inputs=vgg16.input, outputs=predictions)

# 配置 loss funtion 和 optimizer
model.compile(loss='mse', optimizer='SGD')
print(model.summary())
