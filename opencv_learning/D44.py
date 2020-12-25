import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, History
import imgaug.augmenters.flip as aug

"""Working directory: CupoyLearning"""


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


# 回傳定義好的 model 的函數
def buildModel():
    # 定義人臉關鍵點檢測網路
    model = Sequential()

    # 定義神經網路的輸入
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # 最後輸出 30 維的向量，也就是 15 個關鍵點的值
    model.add(Dense(30))
    return model


imgs_train, points_train = loadData(file_name="training")
print("圖像資料:", imgs_train.shape)
print("關鍵點資料:", points_train.shape)

model = buildModel()

# 配置 loss funtion 和 optimizer
model.compile(loss='mse', optimizer='adam')
# print(model.summary())
"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 94, 94, 16)        160       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 47, 47, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 45, 45, 32)        4640      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 22, 22, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 20, 20, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 10, 10, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1049088   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 30)                15390     
=================================================================
Total params: 1,424,286
Trainable params: 1,424,286
Non-trainable params: 0
_________________________________________________________________
None
"""

file_name = 'data/facial-keypoints-detection/best_weights.h5'
checkpoint = ModelCheckpoint(file_name, verbose=2, save_best_only=True)
hist = History()

# training the model
hist_model = model.fit(imgs_train.reshape(-1, 96, 96, 1),
                       points_train,
                       validation_split=0.2, batch_size=64, callbacks=[checkpoint, hist],
                       shuffle=True, epochs=150, verbose=2)
# save the model weights
model.save_weights('data/facial-keypoints-detection/weights.h5')
# save the model
model.save('data/facial-keypoints-detection/model.h5')

# loss 值的圖
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist_model.history['loss'], color='b', label='Training Loss')
plt.plot(hist_model.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

# 讀取測試資料集
imgs_test, _ = loadData(file_name='test')


# 在灰階圖像上畫關鍵點的函數
def plot_keypoints(img, points):
    plt.imshow(img, cmap='gray')
    for i in range(0, 30, 2):
        plt.scatter((points[i] + 0.5) * 96, (points[i + 1] + 0.5) * 96, color='red')


fig = plt.figure(figsize=(15,15))

# 在測試集圖片上用剛剛訓練好的模型做關鍵點的預測
points_test = model.predict(imgs_test.reshape(imgs_test.shape[0], 96, 96, 1))

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(imgs_test[i], np.squeeze(points_test[i]))


"""
作業

請嘗試使用 flip (左右翻轉) 來做 augmentation 以降低人臉關鍵點檢測的 loss

Note: 圖像 flip 之後，groundtruth 的關鍵點也要跟著 flip 哦
"""


def flip(img, point):
    img = aug.fliplr(img)
    length = len(point)

    idxs = np.arange(0, length, 2)
    point[idxs] = -point[idxs]

    return img, point


# train
flip_imgs_train = imgs_train.copy()
flip_points_train = points_train.copy()
n_data = len(flip_imgs_train)

for i in range(n_data):
    flip_imgs_train[i], flip_points_train[i] = flip(flip_imgs_train[i], flip_points_train[i])

file_name = 'data/facial-keypoints-detection/best_flip_weights.h5'
checkpoint = ModelCheckpoint(file_name, verbose=2, save_best_only=True)
hist = History()

# training the model
hist_model = model.fit(flip_imgs_train.reshape(-1, 96, 96, 1),
                       flip_points_train,
                       validation_split=0.2, batch_size=64, callbacks=[checkpoint, hist],
                       shuffle=True, epochs=150, verbose=2)
# save the model weights
model.save_weights('data/facial-keypoints-detection/flip_weights.h5')
# save the model
model.save('data/facial-keypoints-detection/flip_model.h5')

# loss 值的圖
plt.title('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist_model.history['loss'], color='b', label='Training Loss')
plt.plot(hist_model.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')


# test
flip_imgs_test = imgs_test.copy()
flip_points_test = points_test.copy()
n_data = len(flip_imgs_test)

for i in range(n_data):
    flip_imgs_test[i], flip_points_test[i] = flip(flip_imgs_test[i], flip_points_test[i])

fig = plt.figure(figsize=(15,15))

# 在測試集圖片上用剛剛訓練好的模型做關鍵點的預測
flip_points_test_hat = model.predict(flip_imgs_test.reshape(flip_imgs_test.shape[0], 96, 96, 1))

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(imgs_test[i], np.squeeze(flip_points_test_hat[i]))
