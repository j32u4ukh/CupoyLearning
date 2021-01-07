import math

from keras.layers import Convolution2D, Input
from keras.models import Model

"""Working directory: CupoyLearning

運用 Keras 搭建簡單的 Convolution2D Layer，調整 Strides 與 Padding 參數計算輸出 feature map 大小。

填充 (Padding)

『SAME』
 out_height = ceil(float(in_height) / float(strides[1])) 
out_width = ceil(float(in_width) / float(strides[2]))

『Valid』
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))  
out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

輸出 Feature map 尺寸

output = floor((input + 2 * padding - kernel_size) / stride) + 1
"""


def computeFeatureSize(input_shape, padding, kernel_size, stride):
    height, width = input_shape[:2]
    h_padding, w_padding = padding
    h_kernel, w_kernel = kernel_size
    h_stride, w_stride = stride

    h_output = math.floor((height + 2 * h_padding - h_kernel) / h_stride) + 1
    w_output = math.floor((width + 2 * w_padding - w_kernel) / w_stride) + 1

    return h_output, w_output


# kernel size = (6, 6)
# kernel 數量：32

# Same padding、strides=(1,1)
input_shape = (13, 13, 1)
padding = (2, 2)
kernel_size = (5, 5)
stride = (1, 1)
inputs = Input(shape=input_shape)
x = Convolution2D(filters=32, kernel_size=kernel_size, padding="SAME", strides=stride)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())
feature_size = computeFeatureSize(input_shape=input_shape, padding=padding, kernel_size=kernel_size, stride=stride)
print(feature_size)

#  Same padding、strides=(2, 2)
stride = (2, 2)
inputs = Input(shape=input_shape)
x = Convolution2D(filters=32, kernel_size=kernel_size, padding="SAME", strides=(2, 2))(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())
feature_size = computeFeatureSize(input_shape=input_shape, padding=padding, kernel_size=kernel_size, stride=stride)
print(feature_size)

#  Valid padding、strides=(1,1)
stride = (1, 1)
padding = (0, 0)
inputs = Input(shape=input_shape)
x = Convolution2D(filters=32, kernel_size=kernel_size, padding="VALID", strides=stride)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())
feature_size = computeFeatureSize(input_shape=input_shape, padding=padding, kernel_size=kernel_size, stride=stride)
print(feature_size)

#  Valid padding、strides=(2,2)
stride = (2, 2)
inputs = Input(shape=input_shape)
x = Convolution2D(filters=32, kernel_size=kernel_size, padding="VALID", strides=stride)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())
feature_size = computeFeatureSize(input_shape=input_shape, padding=padding, kernel_size=kernel_size, stride=stride)
print(feature_size)
