import math

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential

"""Working directory: CupoyLearning

『目標』: 了解 Maxpooling 的原理與 CNN、FC 層連結的方式
"""


def computeFeatureSize(input_shape, padding, kernel_size, stride):
    height, width = input_shape[:2]
    h_padding, w_padding = padding
    h_kernel, w_kernel = kernel_size
    h_stride, w_stride = stride

    h_output = math.floor((height + 2 * h_padding - h_kernel) / h_stride) + 1
    w_output = math.floor((width + 2 * w_padding - w_kernel) / w_stride) + 1

    return h_output, w_output


input_shape = (32, 32, 3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
print(computeFeatureSize(input_shape=input_shape, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1)))
# (32, 32, 32)

model.add(MaxPooling2D())
# (16, 16, 32)

input_shape = (16, 16, 32)

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
print(computeFeatureSize(input_shape=input_shape, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1)))
# (16, 16, 64)

model.add(MaxPooling2D())
# (8, 8, 64)

input_shape = (8, 8, 64)

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
print(computeFeatureSize(input_shape=input_shape, padding=(1, 1), kernel_size=(3, 3), stride=(1, 1)))
# (8, 8, 128)

model.add(MaxPooling2D())
# (4, 4, 128)

model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
# (4, 4, 10)

# Flatten完尺寸如何變化？
model.add(Flatten())
# (160,)

"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 10)          11530     
_________________________________________________________________
flatten_1 (Flatten)          (None, 160)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 28)                4508      
=================================================================
Total params: 109,286
Trainable params: 109,286
Non-trainable params: 0
"""

# 關掉Flatten，使用GlobalAveragePooling2D，完尺寸如何變化？
model.add(GlobalAveragePooling2D())
# (None, 10)
"""
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 10)          11530     
_________________________________________________________________
global_average_pooling2d_1 ( (None, 10)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 28)                308       
=================================================================
Total params: 105,086
Trainable params: 105,086
Non-trainable params: 0
"""

# 全連接層使用28個units
model.add(Dense(28))
print(model.summary())
