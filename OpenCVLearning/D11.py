from keras.layers import Convolution2D, Input, Dense
from keras.models import Model, Sequential

"""Working directory: CupoyLearning

運用 Keras 搭建簡單的 Dense Layer 與 Convolution2D Layer，使用相同 Neurons 數量，計算總參數量相差多少。
* 輸入照片尺寸: 28 * 28 * 1
* 都用一層，288 個神經元
"""
# 建造一個一層的 FC 層
# 輸入為 28*28*1 攤平 = 784
inputs = Input(shape=(784,))

# CNN 中用了(3*3*1)*32個神經元，我們這邊也用相同神經元數量
x = Dense(288)(inputs)
model = Model(inputs=inputs, outputs=x)

print(model.summary())
"""Total params: 226,080 = (784 + 1) * 288"""

# 建造一個一層的 CNN 層
classifier = Sequential()

# Kernel size 3*3，用 32 張，輸入大小 28 * 28 * 1
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))

"""Total params: 320 = (3 * 3 + 1) * 32"""
print(classifier.summary())



