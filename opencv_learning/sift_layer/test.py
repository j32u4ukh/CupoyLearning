import numpy as np
from keras import layers
from keras.layers import (
    DepthwiseConv2D,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D
)
from keras.layers import Softmax
from keras.models import Input, Model

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


input_tensor = Input((64, 64, 1))
output = layerSiftKeyPoints(input_tensor)
model = Model(inputs=input_tensor, outputs=output)
print(model.summary())
# output.shape = (None, 64, 64, 10)

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

x = np.random.rand(1, 64, 64, 1)
y = np.random.rand(1, 64, 64, 10)

history = model.fit(x, y)

y_hat = model.predict(x)
print(y_hat.shape)
# (1, 64, 64, 10)

input_data = np.random.randint(low=0, high=10, size=(1, 3, 3, 4))
print("input_data")
print(input_data)

input_tensor = Input((3, 3, 4))
output = Softmax(axis=1, input_shape=input_tensor.shape)(input_tensor)
model = Model(inputs=input_tensor, outputs=output)
# print(model.summary())

y_hat = model.predict(input_data)
print("y_hat")
print(y_hat)


