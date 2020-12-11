from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
K.tensorflow_backend._get_available_gpus()


"""Working directory: CupoyLearning

搭建一個 CNN 分類器
"""


# Normalize Data
def normalize(train_data, test_data):
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / (std + 1e-8)
    test_data = (test_data - mean) / (std + 1e-8)
    return train_data, test_data, mean, std


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)  # (50000, 32, 32, 3)

# TODO: 參考 https://keras.io/zh/examples/cifar10_cnn/
# TODO: 訓練集劃分出  Validation 驗證集，來對架構與參數做調整
# Normalize Training and Testset
x_train_norm, x_test_norm, mean_train, std_train = normalize(x_train, x_test)

encoder = OneHotEncoder(sparse=False)
y_train_onehot, y_test_onehot = encoder.fit_transform(y_train), encoder.fit_transform(y_test)

model = Sequential()

# 卷積組合
model.add(Convolution2D(32, kernel_size=(3, 3), padding='same', activation="relu"))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

'''自己決定 MaxPooling2D 放在哪裡'''
model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積組合
model.add(Convolution2D(8, kernel_size=(3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())

# flatten
model.add(Flatten())

# FC
model.add(Dense(activation="relu", units=100))

# 輸出
model.add(Dense(activation="softmax", units=10))

# 超過兩個就要選 categorical_crossentrophy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_norm, y_train_onehot, batch_size=100, epochs=100)

# y_hat = model.predict(x_test_norm)
scores = model.evaluate(x_test_norm, y_test_onehot)
print(scores)
