from keras.layers import Input, Conv2D, MaxPooling2D

"""Working directory: CupoyLearning"""

input_shape_img = (1024, 1024, 3)
img_input = Input(shape=input_shape_img)

'''先過一般 CNN 層提取特徵'''


def getFeaturesCNN(img_input):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # 縮水 1/2 1024x1024 -> 512x512
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # 縮水 1/2 512x512 -> 256x256
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # 縮水 1/2 256x256 -> 128x128
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # 縮水 1/2 128x128 -> 64x64
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    # 最後返回的 x 是 64x64x512 的 feature map。
    return x


def rpnLayer(base_layers, num_anchors):
    x = Conv2D(512, (3, 3),
               padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    # rpn分類和迴歸
    x_class = Conv2D(num_anchors * 2, (1, 1), activation='softmax', name='rpn_out_class')(x)
    x_reg = Conv2D(num_anchors * 4, (1, 1), activation='linear', name='rpn_out_regress')(x)

    return x_class, x_reg, base_layers


base_layers = getFeaturesCNN(img_input)
x_class, x_reg, base_layers = rpnLayer(base_layers, num_anchors=9)

print('Classification支線：', x_class)
# Classification支線： Tensor("rpn_out_class/truediv:0", shape=(?, 64, 64, 18), dtype=float32)

print('BBOX Regression 支線：', x_reg)
# BBOX Regression 支線： Tensor("rpn_out_regress/BiasAdd:0", shape=(?, 64, 64, 36), dtype=float32)

print('CNN Output：', base_layers)
# CNN Output： Tensor("block5_conv3/Relu:0", shape=(?, 64, 64, 512), dtype=float32)
