import numpy as np
import math

from keras import Model
from keras.layers import Convolution2D, Reshape, Dense, Dropout
from keras.layers import Input
from utils.dl import VGG16


n_class = 10
input_shape = (224, 224, 3)
input_tensor = Input(input_shape)

vgg16 = VGG16(include_top=False, input_shape=input_shape, pooling='max')
middle_model = Model(inputs=vgg16.layers[0].input, outputs=vgg16.layers[10].output)
x = middle_model(input_tensor)
conv_shape = x.get_shape()
# print(conv_shape)

# 從(Batch_size,輸出高度,輸出寬度,輸出深度)變成(Batch_size,輸出寬度,輸出深度*輸出高度)，以符合ctc loss需求
x = Reshape(target_shape=(int(conv_shape[2]), int(conv_shape[1] * conv_shape[3])))(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.25)(x)
x = Dense(n_class, activation='softmax')(x)

# 包裝用來預測的model
base_model = Model(inputs=input_tensor, outputs=x)
print(base_model.summary())
