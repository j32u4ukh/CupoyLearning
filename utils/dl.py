import math
import os
import numpy as np
from keras.datasets.cifar import load_batch
from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers import (
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D
)
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Add
from keras.models import Input
from keras.models import Model


def computeFeatureSize(input_shape, padding, kernel_size, stride):
    """
    計算進行卷積運算後的數據大小

    填充 (Padding)
    『SAME』
     out_height = ceil(float(in_height) / float(strides[1])) 
    out_width = ceil(float(in_width) / float(strides[2]))

    『Valid』
    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))  
    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    輸出 Feature map 尺寸
    output = floor((input + 2 * padding - kernel_size) / stride) + 1

    :param input_shape:
    :param padding:
    :param kernel_size:
    :param stride:
    :return:
    """
    height, width = input_shape[:2]
    h_padding, w_padding = padding
    h_kernel, w_kernel = kernel_size
    h_stride, w_stride = stride

    h_output = math.floor((height + 2 * h_padding - h_kernel) / h_stride) + 1
    w_output = math.floor((width + 2 * w_padding - w_kernel) / w_stride) + 1

    return h_output, w_output


def VGG16(include_top=True, input_shape=(224, 224, 1), pooling='max', classes=1000):
    input_data = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
    # (None, 224, 224, 64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # (None, 224, 224, 64)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # (None, 112, 112, 64)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # (None, 112, 112, 128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # (None, 112, 112, 128)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # (None, 56, 56, 128)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # (None, 56, 56, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # (None, 56, 56, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # (None, 56, 56, 256)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # (None, 28, 28, 256)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # (None, 28, 28, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # (None, 28, 28, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # (None, 28, 28, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # (None, 14, 14, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # (None, 14, 14, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # (None, 14, 14, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # (None, 7, 7, 512)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='多元分類的輸出層', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            # (None, 512)

    # Create model.
    model = Model(input_data, x, name='vgg16')

    return model


def batchNormalizationConv2D(x, filters, kernel_size, padding='same', strides=(1, 1), normalizer=True,
                             activation='relu', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = Conv2D(filters, kernel_size,
               strides=strides, padding=padding,
               use_bias=False, name=conv_name)(x)

    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if activation:
        x = Activation(activation, name=act_name)(x)

    return x


def blockInceptionV1(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs  # ((64,), (96,128), (16,32), (32,))

    '''Branch_0'''
    branch_0 = batchNormalizationConv2D(x, br0[0], (1, 1), name=name + "_Branch_0")

    '''Branch_1'''
    branch_1 = batchNormalizationConv2D(x, br1[0], (1, 1), name=name + "_Branch_1")
    branch_1 = batchNormalizationConv2D(branch_1, br1[1], (3, 3), name=name + "_Branch_1_1")

    '''Branch_2'''
    branch_2 = batchNormalizationConv2D(x, br2[0], (1, 1), name=name + "_Branch_2")
    branch_2 = batchNormalizationConv2D(branch_2, br2[1], (5, 5), name=name + "_Branch_2_1")

    '''Branch_3'''
    branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=name + '_Branch_3')(x)
    branch_3 = batchNormalizationConv2D(branch_3, br3[0], (1, 1), name=name + "_Branch_3_1")

    x = layers.concatenate([branch_0, branch_1, branch_2, branch_3],
                           axis=channel_axis,
                           name=name + "_Concatenated")

    return x


def blockInceptionV3(x, specs, filter_size, channel_axis, name):
    # specs: ((64,), (32,), (8, 16, 32), (8, 16, 32, 64, 128))
    (br0, br1, br2, br3) = specs

    '''Branch_0'''
    branch_0 = batchNormalizationConv2D(x, br0[0], (1, 1), name=name + "_Branch_0")

    # branch_0: (?, 224, 224, 64)
    # print("branch_0:", branch_0.shape)

    '''Branch_1'''
    branch_1 = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=name + '_Branch_1')(x)
    branch_1 = batchNormalizationConv2D(branch_1, br1[0], (1, 1), name=name + "_Branch_1_1")

    # branch_1: (?, 224, 224, 32)
    # print("branch_1:", branch_1.shape)

    '''Branch_2'''
    branch_2 = batchNormalizationConv2D(x, br2[0], (1, 1), name=name + "_Branch_2")
    branch_2 = batchNormalizationConv2D(branch_2, br2[1], (1, filter_size), name=name + "_Branch_2_1")
    branch_2 = batchNormalizationConv2D(branch_2, br2[2], (filter_size, 1), name=name + "_Branch_2_2")

    # branch_2: (?, 224, 224, 32)
    # print("branch_2:", branch_2.shape)

    '''Branch_3'''
    branch_3 = batchNormalizationConv2D(x, br3[0], (1, 1), name=name + "_Branch_3")
    branch_3 = batchNormalizationConv2D(branch_3, br3[1], (1, filter_size), name=name + "_Branch_3_1")
    branch_3 = batchNormalizationConv2D(branch_3, br3[2], (filter_size, 1), name=name + "_Branch_3_2")
    branch_3 = batchNormalizationConv2D(branch_3, br3[3], (1, filter_size), name=name + "_Branch_3_3")
    branch_3 = batchNormalizationConv2D(branch_3, br3[4], (filter_size, 1), name=name + "_Branch_3_4")

    # branch_3: (?, 224, 224, 128)
    # print("branch_3:", branch_3.shape)

    x = layers.concatenate([branch_0, branch_1, branch_2, branch_3],
                           axis=channel_axis,
                           name=name + "_Concatenated")

    # x: (?, 224, 224, 256)
    # print("x:", x.shape)

    return x


def inceptionVGG16(include_top=True, input_shape=(224, 224, 1), pooling='max', classes=1000):
    # 所有 Convolution 改為 BatchNormalizationConv2D
    # batchNormalizationConv2D(x, filters, kernel_size, padding='same', strides=(1, 1), normalizer=True,
    #                              activation='relu', name=None)
    input_data = Input(shape=input_shape)

    # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
    x = batchNormalizationConv2D(input_data, 64, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block1_conv1')
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = batchNormalizationConv2D(x, 64, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = batchNormalizationConv2D(x, 128, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block2_conv1')
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = batchNormalizationConv2D(x, 128, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3: blockInceptionV1(input_data, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_1')
    # Block_3 中的 Convolution 全部改為 blockInceptionV1
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'block3_conv1')
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'block3_conv2')
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'block3_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = batchNormalizationConv2D(x, 512, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block4_conv1')
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = batchNormalizationConv2D(x, 512, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block4_conv2')
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = batchNormalizationConv2D(x, 512, (3, 3), padding='same', strides=(1, 1), normalizer=True,
                                 activation='relu', name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # Block_5 中的 Convolution 全部改為 blockInceptionV3
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)),
                         filter_size=3, channel_axis=3, name='block5_conv1')
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)),
                         filter_size=3, channel_axis=3, name='block5_conv2')
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)),
                         filter_size=3, channel_axis=3, name='block5_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='多元分類的輸出層', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(input_data, x, name='InceptionVGG16')

    return model


def residualBlockV1(input_data, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = f'res_{stage}_{block}_branch'
    bn_name_base = f'bn_{stage}_{block}_branch'

    x = Conv2D(filter1, (1, 1), padding='same', name=conv_name_base + '1')(input_data)
    x = BatchNormalization(axis=3, name=bn_name_base + '2')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '3')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '4')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), padding='same', name=conv_name_base + '5')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '6')(x)

    x = layers.add([x, input_data])
    x = Activation('relu')(x)
    return x


"""參考 ResNetV1 搭建 ResNetV2 版本的 Residual Block"""


def residualBlockV2(input_data, kernel_size, filters, stage, block):
    filter1, filter2 = filters

    assert filter2 == input_data.shape[-1], "輸入與輸出Feature Map必須有相同大小跟深度，不然無法直接相加。"

    conv_name_base = f'res_{stage}_{block}_branch'
    bn_name_base = f'bn_{stage}_{block}_branch'

    x = BatchNormalization(axis=3, name=bn_name_base + '1')(input_data)
    x = Activation('relu')(x)
    x = Conv2D(filter1, (1, 1), padding='same', name=conv_name_base + '2')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '3')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '4')(x)

    x = layers.add([x, input_data])
    x = Activation('relu')(x)
    return x


def residualZipBlockV2(input_data, kernel_size, stage, block, reduce=96, ouput_size=128):
    conv_name_base = f'res_{stage}_{block}_branch'
    bn_name_base = f'bn_{stage}_{block}_branch'

    x = BatchNormalization(axis=3, name=bn_name_base + '1')(input_data)
    x = Activation('relu')(x)
    x = Conv2D(reduce, (1, 1), padding='same', name=conv_name_base + '2')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '3')(x)
    x = Activation('relu')(x)
    x = Conv2D(ouput_size, kernel_size, padding='same', name=conv_name_base + '4')(x)

    x = layers.add([x, input_data])
    x = Activation('relu')(x)
    return x


def layerInvertedResidual(input_tensor, expansion):
    """
    嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Inverted Residual Block.

    :param input_tensor:
    :param expansion: expand filters size
    :return:
    """

    x = DepthwiseConv2D(3, padding="same", depth_multiplier=expansion, input_shape=input_tensor.shape)(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = Conv2D(3, kernel_size=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    added = Add()([input_tensor, x])

    '''
    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_7 (InputLayer)            (None, 64, 64, 3)    0                                            
    __________________________________________________________________________________________________
    depthwise_conv2d_5 (DepthwiseCo (None, 64, 64, 18)   180         input_7[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 64, 64, 18)   256         depthwise_conv2d_5[0][0]         
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 64, 64, 18)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 64, 64, 3)    57          activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 64, 64, 3)    256         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 64, 64, 3)    0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 64, 64, 3)    0           input_7[0][0]                    
                                                                     activation_4[0][0]               
    ==================================================================================================
    Total params: 749
    Trainable params: 493
    Non-trainable params: 256
    __________________________________________________________________________________________________
    None
    '''

    return added


def layerSeparableConv(input_tensor):
    """
    嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Separable Convolution.

    :param input_tensor:
    :return:
    """
    x = DepthwiseConv2D(3, padding="same", input_shape=input_tensor.shape)(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)

    '''Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_6 (InputLayer)         (None, 64, 64, 3)         0         
    _________________________________________________________________
    depthwise_conv2d_4 (Depthwis (None, 64, 64, 3)         30        
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 64, 64, 3)         256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 64, 64, 3)         0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 64, 64, 128)       512       
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 64, 64, 128)       256       
    _________________________________________________________________
    activation_2 (Activation)    (None, 64, 64, 128)       0         
    =================================================================
    Total params: 1,054
    Trainable params: 798
    Non-trainable params: 256
    _________________________________________________________________
    None
    '''

    return x


def loadCifar10():
    path = r"C:\Users\user\.keras\datasets\cifar-10-batches-py"

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose([0, 2, 3, 1])
        x_test = x_test.transpose([0, 2, 3, 1])

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # model = VGG16(include_top=False, input_shape=(224, 224, 1), pooling='max', classes=1000)
    # layer = model.layers[10]
    # print(layer.output)
    (x_train, y_train), (x_test, y_test) = loadCifar10()
