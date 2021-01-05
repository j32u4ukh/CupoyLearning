"""
特徵值計算需要 16 X 16 的大小
"""
import os
import pickle
from keras import backend as K
import numpy as np
from keras import layers
from keras.layers import (
    DepthwiseConv2D,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D
)
from keras.models import Input, Model
from matplotlib import pyplot as plt

from utils.dl import loadCifar10
from utils.opencv import siftDetect

np.set_printoptions(precision=4, suppress=True)


def layerSiftKeyPoints(input_tensor, n_class=10, kernel_n=3):
    input_shape = input_tensor.shape
    x1 = DepthwiseConv2D(kernel_size=3, padding="same", name='b1_dc1', input_shape=input_shape)(input_tensor)
    x1 = Conv2D(n_class, kernel_size=(1, 1), padding="same", name='b1_c1')(x1)
    x1 = BatchNormalization(axis=3, name='b1_bn1')(x1)
    x1 = Activation('relu', name='b1_a1')(x1)

    x2 = DepthwiseConv2D(kernel_size=3, padding="same", name='b2_dc1', input_shape=input_shape)(input_tensor)
    x2 = Conv2D(n_class, kernel_size=(1, 1), padding="same", name='b2_c1')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same", name='b2_mp1')(x2)
    x2 = BatchNormalization(axis=3, name='b2_bn1')(x2)
    x2 = Activation('relu', name='b2_a1')(x2)

    x3 = DepthwiseConv2D(kernel_size=3, padding="same", name='b3_dc1', input_shape=input_shape)(input_tensor)
    x3 = Conv2D(n_class, kernel_size=(1, 1), padding="same", name='b3_c1')(x3)
    x3 = Conv2D(n_class, kernel_size=(1, kernel_n), padding="same", name='b3_c2')(x3)
    x3 = Conv2D(n_class, kernel_size=(kernel_n, 1), padding="same", name='b3_c3')(x3)
    x3 = BatchNormalization(axis=3, name='b3_bn1')(x3)
    x3 = Activation('relu', name='b3_a1')(x3)

    x4 = DepthwiseConv2D(kernel_size=3, padding="same", name='b4_dc1', input_shape=input_shape)(input_tensor)
    x4 = Conv2D(n_class, kernel_size=(1, 1), padding="same", name='b4_c1')(x4)
    x4 = Conv2D(n_class, kernel_size=(1, kernel_n), padding="same", name='b4_c2')(x4)
    x4 = Conv2D(n_class, kernel_size=(kernel_n, 1), padding="same", name='b4_c3')(x4)
    x4 = Conv2D(n_class, kernel_size=(1, kernel_n), padding="same", name='b4_c4')(x4)
    x4 = Conv2D(n_class, kernel_size=(kernel_n, 1), padding="same", name='b4_c5')(x4)
    x4 = BatchNormalization(axis=3, name='b4_bn1')(x4)
    x4 = Activation('relu', name='b4_a1')(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=3)
    x = Conv2D(n_class, kernel_size=1, padding="same", name='concat_c1')(x)
    x = BatchNormalization(axis=3, name='concat_bn1')(x)
    x = Activation('softmax', name='concat_a1')(x)

    return x


def computeSiftKeyPoints(img, label, n_class=10, idx=0):
    keypoints = siftDetect(img)

    height, width, _ = img.shape
    points = np.zeros((height, width, n_class))

    if keypoints is not None:
        n_keypoint = len(keypoints)

        for i in range(n_keypoint):
            keypoint = keypoints[i]

            pt = keypoint.pt
            points[round(pt[0]), round(pt[1]), label] = 1.0

        return points
    else:
        print(f"Image ({idx}) is None.")
        return []


def produceSiftData(datas, labels, file_idx=0):
    path = os.path.join("data", f"batch{file_idx}.pickle")
    start = 10000 * file_idx
    stop = 10000 * (file_idx + 1)

    # TODO: 應改為多核心或多執行續
    indexs, points_list = [], []

    for i in range(start, stop):
        data = datas[i]
        label = labels[i][0]
        points = computeSiftKeyPoints(img=data, label=label, idx=i)

        if len(points) > 0:
            indexs.append(i)
            points_list.append(points)

    indexs = np.array(indexs)
    points_list = np.array(points_list)
    sift_data = (datas[indexs], points_list)

    with open(path, "wb") as f:
        pickle.dump(sift_data, f)


def loadSiftData(file_idx=0):
    path = os.path.join("data", f"batch{file_idx}.pickle")

    with open(path, "rb") as f:
        datas = pickle.load(f)
        indexs, points_list = datas
        return indexs, points_list


cifar10_labels = {
    0: "airplain",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

(x_train, y_train), (x_test, y_test) = loadCifar10()
print("x_train:", x_train.shape, ", y_train:", y_train.shape)
# x_train: (50000, 32, 32, 3) , y_train: (50000, 1)
# -> file_idx: 0 - 4
# TODO: 直接根據 indexs 將相對應 image 和 points_list 一起儲存
# produceSiftData(x_train, y_train, file_idx=4)
indexs, points_list = loadSiftData(file_idx=4)

# small_indexs = indexs[:100]
small_x_train = x_train[indexs]
# small_points_list = points_list[:100]

input_tensor = Input((32, 32, 3))
output = layerSiftKeyPoints(input_tensor)
model = Model(inputs=input_tensor, outputs=output)
print(model.summary())

model.load_weights('data/sift_points_model.h5')
# output.shape = (None, 64, 64, 10)
model.compile(optimizer="sgd", loss='categorical_crossentropy',
              metrics=['acc'])

batch_size = 50
epochs = 20
history = model.fit(x=small_x_train, y=points_list,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=2)

# TODO: 降低暫存記憶體使用量
K.clear_session()

# 繪制訓練 & 驗證的準確率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 繪制訓練 & 驗證的損失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 將模型儲存至 HDF5 檔案中
model.save_weights('data/sift_points_model.h5')
