import warnings

import keras
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras
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
print(x_train.shape)


# Normalize Training and Testset
scaler = StandardScaler()
x_train_norm = scaler.fitTransform(x_train)
x_test_norm = scaler.transform(x_test)

# label to onehot encodeing
y_train_onehot = keras.utils.to_categorical(y_train, n_class)
y_test_onehot = keras.utils.to_categorical(y_test, n_class)

# 劃分'訓練集'與'驗證集'，來對架構與超參數做調整
x_train_norm, x_validation_norm, y_train_onehot, y_validation_onehot = train_test_split(x_train_norm, y_train_onehot,
                                                                                        test_size=0.3,
                                                                                        random_state=seed)
# x_train_norm.shape: (35000, 32, 32, 3), y_train_onehot.shape: (35000, 10)
# x_validation_norm.shape: (15000, 32, 32, 3), y_validation_onehot.shape: (15000, 10)

# region 參考 Keras 文檔 https://keras.io/zh/examples/cifar10_cnn/
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

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
model.fit(x_train_norm, y_train_onehot,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_validation_norm, y_validation_onehot),
          shuffle=True)

"""模型架構與超參數的調整在驗證集(Validation)獲得良好表現後，才利用測試集衡量表現"""
# y_hat = model.predict(x_test_norm)
scores = model.evaluate(x_test_norm, y_test_onehot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
