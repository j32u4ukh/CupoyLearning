import warnings

import keras
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.opencv import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
K.tensorflow_backend._get_available_gpus()

"""Working directory: CupoyLearning

搭建一個 CNN 分類器
"""

seed = 7
np.random.seed(seed)

batch_size = 32
n_class = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# (50000, 32, 32, 3)
print(x_train.shape)

# Normalize Training and Testset
scaler = StandardScaler()
x_train_norm = scaler.fitTransform(x_train)
x_test_norm = scaler.transform(x_test)

# label to onehot encodeing
y_train_onehot = keras.utils.to_categorical(y_train, n_class)
y_test_onehot = keras.utils.to_categorical(y_test, n_class)

# 劃分'訓練集'與'驗證集'，來對架構與超參數做調整
# 如果數據本身是有序的，需要先 shuffle 再劃分 validation，否則可能會出現驗證集樣本不均勻。
indexs = list(range(len(x_train_norm)))
np.random.shuffle(indexs)
x_train_norm = x_train_norm[indexs]
y_train_onehot = y_train_onehot[indexs]
x_train_norm, x_validation_norm, y_train_onehot, y_validation_onehot = train_test_split(x_train_norm, y_train_onehot,
                                                                                        test_size=0.3,
                                                                                        random_state=seed)

# x_train_norm.shape: (35000, 32, 32, 3), y_train_onehot.shape: (35000, 10)
# x_validation_norm.shape: (15000, 32, 32, 3), y_validation_onehot.shape: (15000, 10)
# x_test_norm.shape: (10000, 32, 32, 3), y_test_onehot.shape: (10000, 10)
print(f"x_train_norm.shape: {x_train_norm.shape}, y_train_onehot.shape: {y_train_onehot.shape}")
print(f"x_validation_norm.shape: {x_validation_norm.shape}, y_validation_onehot.shape: {y_validation_onehot.shape}")
print(f"x_test_norm.shape: {x_test_norm.shape}, y_test_onehot.shape: {y_test_onehot.shape}")

input_shape = x_train_norm.shape[1:]

# input_shape: (32, 32, 3)
print("input_shape:", input_shape)

# region 參考 Keras 文檔 https://keras.io/zh/examples/cifar10_cnn/
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# model.add(GlobalAveragePooling2D())

model.add(Dense(512))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
model.add(Dropout(0.5))

model.add(Dense(n_class))
model.add(Activation('softmax'))
# endregion

# 超過兩個就要選 categorical_crossentrophy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_norm, y_train_onehot,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_validation_norm, y_validation_onehot),
                    shuffle=True,
                    verbose=2)

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

""" 模型架構與超參數的調整在驗證集(Validation)獲得良好表現後，才利用測試集衡量表現 """
# y_hat = model.predict(x_test_norm)
scores = model.evaluate(x_test_norm, y_test_onehot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
