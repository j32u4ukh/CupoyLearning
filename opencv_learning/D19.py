import warnings

from keras import backend as K
from keras import layers
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    MaxPooling2D
)
from keras.models import Model

warnings.simplefilter(action='ignore', category=FutureWarning)

"""Working directory: CupoyLearning

『本次練習內容』: 學習如何搭建Inception Block

『本次練習目的』: 了解Inceotion原理 & 了解如何導入 Inception block 到原本架構中
"""


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


def VGG16(include_top=True, input_shape=(224, 224, 1), pooling='max', classes=1000):
    input_data = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
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
    model = Model(input_data, x, name='vgg16')

    return model


def testBlockInceptionV1():
    input_data = Input(shape=(224, 224, 1))
    x = blockInceptionV1(input_data, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_1')

    # Tensor("Block_1_Concatenated/concat:0", shape=(?, 224, 224, 256), dtype=float32)
    print(x)


"""
額外練習

將 VGG16 Block_3 中的 Convolution 全部改為 blockInceptionV1
Block_5 中的 Convolution 全部改為 blockInceptionV3
並將所有 Convolution 改為 BatchNormalizationConv2D
"""


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


def testBlockInceptionV3():
    input_data = Input(shape=(224, 224, 1))
    x = blockInceptionV3(input_data, ((64,), (32,), (8, 16, 32), (8, 16, 32, 64, 128)),
                         filter_size=3, channel_axis=3, name='Block_1')

    # Tensor("Block_1_Concatenated_1/concat:0", shape=(?, 224, 224, 256), dtype=float32)
    print(x)


def inceptionVGG16(include_top=True, input_shape=(224, 224, 1), pooling='max', classes=1000):
    # TODO: 所有 Convolution 改為 BatchNormalizationConv2D
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
    # TODO: Block_3 中的 Convolution 全部改為 blockInceptionV1
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
    # TODO: Block_5 中的 Convolution 全部改為 blockInceptionV3
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


def testInceptionVGG16():
    model = inceptionVGG16(include_top=False)
    print(model.summary())

    """
    Model: "InceptionVGG16"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_4 (InputLayer)            (None, 224, 224, 1)  0                                            
    __________________________________________________________________________________________________
    block1_conv1_conv (Conv2D)      (None, 224, 224, 64) 576         input_4[0][0]                    
    __________________________________________________________________________________________________
    block1_conv1_bn (BatchNormaliza (None, 224, 224, 64) 192         block1_conv1_conv[0][0]          
    __________________________________________________________________________________________________
    block1_conv1_act (Activation)   (None, 224, 224, 64) 0           block1_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block1_conv2_conv (Conv2D)      (None, 224, 224, 64) 36864       block1_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block1_conv2_bn (BatchNormaliza (None, 224, 224, 64) 192         block1_conv2_conv[0][0]          
    __________________________________________________________________________________________________
    block1_conv2_act (Activation)   (None, 224, 224, 64) 0           block1_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block1_pool (MaxPooling2D)      (None, 112, 112, 64) 0           block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_conv1_conv (Conv2D)      (None, 112, 112, 128 73728       block1_pool[0][0]                
    __________________________________________________________________________________________________
    block2_conv1_bn (BatchNormaliza (None, 112, 112, 128 384         block2_conv1_conv[0][0]          
    __________________________________________________________________________________________________
    block2_conv1_act (Activation)   (None, 112, 112, 128 0           block2_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block2_conv2_conv (Conv2D)      (None, 112, 112, 128 147456      block2_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block2_conv2_bn (BatchNormaliza (None, 112, 112, 128 384         block2_conv2_conv[0][0]          
    __________________________________________________________________________________________________
    block2_conv2_act (Activation)   (None, 112, 112, 128 0           block2_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block2_pool (MaxPooling2D)      (None, 56, 56, 128)  0           block2_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_conv (Con (None, 56, 56, 96)   12288       block2_pool[0][0]                
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_conv (Con (None, 56, 56, 16)   2048        block2_pool[0][0]                
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_bn (Batch (None, 56, 56, 96)   288         block3_conv1_Branch_1_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_bn (Batch (None, 56, 56, 16)   48          block3_conv1_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_act (Acti (None, 56, 56, 96)   0           block3_conv1_Branch_1_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_act (Acti (None, 56, 56, 16)   0           block3_conv1_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv1_Branch_3 (MaxPooli (None, 56, 56, 128)  0           block2_pool[0][0]                
    __________________________________________________________________________________________________
    block3_conv1_Branch_0_conv (Con (None, 56, 56, 64)   8192        block2_pool[0][0]                
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_1_conv (C (None, 56, 56, 128)  110592      block3_conv1_Branch_1_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_1_conv (C (None, 56, 56, 32)   12800       block3_conv1_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv1_Branch_3_1_conv (C (None, 56, 56, 32)   4096        block3_conv1_Branch_3[0][0]      
    __________________________________________________________________________________________________
    block3_conv1_Branch_0_bn (Batch (None, 56, 56, 64)   192         block3_conv1_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_1_bn (Bat (None, 56, 56, 128)  384         block3_conv1_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_1_bn (Bat (None, 56, 56, 32)   96          block3_conv1_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv1_Branch_3_1_bn (Bat (None, 56, 56, 32)   96          block3_conv1_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv1_Branch_0_act (Acti (None, 56, 56, 64)   0           block3_conv1_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv1_Branch_1_1_act (Ac (None, 56, 56, 128)  0           block3_conv1_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Branch_2_1_act (Ac (None, 56, 56, 32)   0           block3_conv1_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Branch_3_1_act (Ac (None, 56, 56, 32)   0           block3_conv1_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv1_Concatenated (Conc (None, 56, 56, 256)  0           block3_conv1_Branch_0_act[0][0]  
                                                                     block3_conv1_Branch_1_1_act[0][0]
                                                                     block3_conv1_Branch_2_1_act[0][0]
                                                                     block3_conv1_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_conv (Con (None, 56, 56, 96)   24576       block3_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_conv (Con (None, 56, 56, 16)   4096        block3_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_bn (Batch (None, 56, 56, 96)   288         block3_conv2_Branch_1_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_bn (Batch (None, 56, 56, 16)   48          block3_conv2_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_act (Acti (None, 56, 56, 96)   0           block3_conv2_Branch_1_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_act (Acti (None, 56, 56, 16)   0           block3_conv2_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv2_Branch_3 (MaxPooli (None, 56, 56, 256)  0           block3_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_0_conv (Con (None, 56, 56, 64)   16384       block3_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_1_conv (C (None, 56, 56, 128)  110592      block3_conv2_Branch_1_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_1_conv (C (None, 56, 56, 32)   12800       block3_conv2_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv2_Branch_3_1_conv (C (None, 56, 56, 32)   8192        block3_conv2_Branch_3[0][0]      
    __________________________________________________________________________________________________
    block3_conv2_Branch_0_bn (Batch (None, 56, 56, 64)   192         block3_conv2_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_1_bn (Bat (None, 56, 56, 128)  384         block3_conv2_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_1_bn (Bat (None, 56, 56, 32)   96          block3_conv2_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv2_Branch_3_1_bn (Bat (None, 56, 56, 32)   96          block3_conv2_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv2_Branch_0_act (Acti (None, 56, 56, 64)   0           block3_conv2_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv2_Branch_1_1_act (Ac (None, 56, 56, 128)  0           block3_conv2_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Branch_2_1_act (Ac (None, 56, 56, 32)   0           block3_conv2_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Branch_3_1_act (Ac (None, 56, 56, 32)   0           block3_conv2_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv2_Concatenated (Conc (None, 56, 56, 256)  0           block3_conv2_Branch_0_act[0][0]  
                                                                     block3_conv2_Branch_1_1_act[0][0]
                                                                     block3_conv2_Branch_2_1_act[0][0]
                                                                     block3_conv2_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_conv (Con (None, 56, 56, 96)   24576       block3_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_conv (Con (None, 56, 56, 16)   4096        block3_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_bn (Batch (None, 56, 56, 96)   288         block3_conv3_Branch_1_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_bn (Batch (None, 56, 56, 16)   48          block3_conv3_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_act (Acti (None, 56, 56, 96)   0           block3_conv3_Branch_1_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_act (Acti (None, 56, 56, 16)   0           block3_conv3_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv3_Branch_3 (MaxPooli (None, 56, 56, 256)  0           block3_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_0_conv (Con (None, 56, 56, 64)   16384       block3_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_1_conv (C (None, 56, 56, 128)  110592      block3_conv3_Branch_1_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_1_conv (C (None, 56, 56, 32)   12800       block3_conv3_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block3_conv3_Branch_3_1_conv (C (None, 56, 56, 32)   8192        block3_conv3_Branch_3[0][0]      
    __________________________________________________________________________________________________
    block3_conv3_Branch_0_bn (Batch (None, 56, 56, 64)   192         block3_conv3_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_1_bn (Bat (None, 56, 56, 128)  384         block3_conv3_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_1_bn (Bat (None, 56, 56, 32)   96          block3_conv3_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv3_Branch_3_1_bn (Bat (None, 56, 56, 32)   96          block3_conv3_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block3_conv3_Branch_0_act (Acti (None, 56, 56, 64)   0           block3_conv3_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block3_conv3_Branch_1_1_act (Ac (None, 56, 56, 128)  0           block3_conv3_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Branch_2_1_act (Ac (None, 56, 56, 32)   0           block3_conv3_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Branch_3_1_act (Ac (None, 56, 56, 32)   0           block3_conv3_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block3_conv3_Concatenated (Conc (None, 56, 56, 256)  0           block3_conv3_Branch_0_act[0][0]  
                                                                     block3_conv3_Branch_1_1_act[0][0]
                                                                     block3_conv3_Branch_2_1_act[0][0]
                                                                     block3_conv3_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block3_pool (MaxPooling2D)      (None, 28, 28, 256)  0           block3_conv3_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block4_conv1_conv (Conv2D)      (None, 28, 28, 512)  1179648     block3_pool[0][0]                
    __________________________________________________________________________________________________
    block4_conv1_bn (BatchNormaliza (None, 28, 28, 512)  1536        block4_conv1_conv[0][0]          
    __________________________________________________________________________________________________
    block4_conv1_act (Activation)   (None, 28, 28, 512)  0           block4_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block4_conv2_conv (Conv2D)      (None, 28, 28, 512)  2359296     block4_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block4_conv2_bn (BatchNormaliza (None, 28, 28, 512)  1536        block4_conv2_conv[0][0]          
    __________________________________________________________________________________________________
    block4_conv2_act (Activation)   (None, 28, 28, 512)  0           block4_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block4_conv3_conv (Conv2D)      (None, 28, 28, 512)  2359296     block4_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block4_conv3_bn (BatchNormaliza (None, 28, 28, 512)  1536        block4_conv3_conv[0][0]          
    __________________________________________________________________________________________________
    block4_conv3_act (Activation)   (None, 28, 28, 512)  0           block4_conv3_bn[0][0]            
    __________________________________________________________________________________________________
    block4_pool (MaxPooling2D)      (None, 14, 14, 512)  0           block4_conv3_act[0][0]           
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_conv (Con (None, 14, 14, 16)   8192        block4_pool[0][0]                
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_bn (Batch (None, 14, 14, 16)   48          block5_conv1_Branch_3_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_act (Acti (None, 14, 14, 16)   0           block5_conv1_Branch_3_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_1_conv (C (None, 14, 14, 32)   1536        block5_conv1_Branch_3_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_1_bn (Bat (None, 14, 14, 32)   96          block5_conv1_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_1_act (Ac (None, 14, 14, 32)   0           block5_conv1_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_conv (Con (None, 14, 14, 16)   8192        block4_pool[0][0]                
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_2_conv (C (None, 14, 14, 64)   6144        block5_conv1_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_bn (Batch (None, 14, 14, 16)   48          block5_conv1_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_2_bn (Bat (None, 14, 14, 64)   192         block5_conv1_Branch_3_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_act (Acti (None, 14, 14, 16)   0           block5_conv1_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_2_act (Ac (None, 14, 14, 64)   0           block5_conv1_Branch_3_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_1_conv (C (None, 14, 14, 32)   1536        block5_conv1_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_3_conv (C (None, 14, 14, 128)  24576       block5_conv1_Branch_3_2_act[0][0]
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_1_bn (Bat (None, 14, 14, 32)   96          block5_conv1_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_3_bn (Bat (None, 14, 14, 128)  384         block5_conv1_Branch_3_3_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_1 (MaxPooli (None, 14, 14, 512)  0           block4_pool[0][0]                
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_1_act (Ac (None, 14, 14, 32)   0           block5_conv1_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_3_act (Ac (None, 14, 14, 128)  0           block5_conv1_Branch_3_3_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_0_conv (Con (None, 14, 14, 128)  65536       block4_pool[0][0]                
    __________________________________________________________________________________________________
    block5_conv1_Branch_1_1_conv (C (None, 14, 14, 64)   32768       block5_conv1_Branch_1[0][0]      
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_2_conv (C (None, 14, 14, 64)   6144        block5_conv1_Branch_2_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_4_conv (C (None, 14, 14, 256)  98304       block5_conv1_Branch_3_3_act[0][0]
    __________________________________________________________________________________________________
    block5_conv1_Branch_0_bn (Batch (None, 14, 14, 128)  384         block5_conv1_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_1_1_bn (Bat (None, 14, 14, 64)   192         block5_conv1_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_2_bn (Bat (None, 14, 14, 64)   192         block5_conv1_Branch_2_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_4_bn (Bat (None, 14, 14, 256)  768         block5_conv1_Branch_3_4_conv[0][0
    __________________________________________________________________________________________________
    block5_conv1_Branch_0_act (Acti (None, 14, 14, 128)  0           block5_conv1_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv1_Branch_1_1_act (Ac (None, 14, 14, 64)   0           block5_conv1_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_2_2_act (Ac (None, 14, 14, 64)   0           block5_conv1_Branch_2_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Branch_3_4_act (Ac (None, 14, 14, 256)  0           block5_conv1_Branch_3_4_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv1_Concatenated (Conc (None, 14, 14, 512)  0           block5_conv1_Branch_0_act[0][0]  
                                                                     block5_conv1_Branch_1_1_act[0][0]
                                                                     block5_conv1_Branch_2_2_act[0][0]
                                                                     block5_conv1_Branch_3_4_act[0][0]
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_conv (Con (None, 14, 14, 16)   8192        block5_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_bn (Batch (None, 14, 14, 16)   48          block5_conv2_Branch_3_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_act (Acti (None, 14, 14, 16)   0           block5_conv2_Branch_3_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_1_conv (C (None, 14, 14, 32)   1536        block5_conv2_Branch_3_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_1_bn (Bat (None, 14, 14, 32)   96          block5_conv2_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_1_act (Ac (None, 14, 14, 32)   0           block5_conv2_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_conv (Con (None, 14, 14, 16)   8192        block5_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_2_conv (C (None, 14, 14, 64)   6144        block5_conv2_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_bn (Batch (None, 14, 14, 16)   48          block5_conv2_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_2_bn (Bat (None, 14, 14, 64)   192         block5_conv2_Branch_3_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_act (Acti (None, 14, 14, 16)   0           block5_conv2_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_2_act (Ac (None, 14, 14, 64)   0           block5_conv2_Branch_3_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_1_conv (C (None, 14, 14, 32)   1536        block5_conv2_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_3_conv (C (None, 14, 14, 128)  24576       block5_conv2_Branch_3_2_act[0][0]
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_1_bn (Bat (None, 14, 14, 32)   96          block5_conv2_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_3_bn (Bat (None, 14, 14, 128)  384         block5_conv2_Branch_3_3_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_1 (MaxPooli (None, 14, 14, 512)  0           block5_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_1_act (Ac (None, 14, 14, 32)   0           block5_conv2_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_3_act (Ac (None, 14, 14, 128)  0           block5_conv2_Branch_3_3_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_0_conv (Con (None, 14, 14, 128)  65536       block5_conv1_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv2_Branch_1_1_conv (C (None, 14, 14, 64)   32768       block5_conv2_Branch_1[0][0]      
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_2_conv (C (None, 14, 14, 64)   6144        block5_conv2_Branch_2_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_4_conv (C (None, 14, 14, 256)  98304       block5_conv2_Branch_3_3_act[0][0]
    __________________________________________________________________________________________________
    block5_conv2_Branch_0_bn (Batch (None, 14, 14, 128)  384         block5_conv2_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_1_1_bn (Bat (None, 14, 14, 64)   192         block5_conv2_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_2_bn (Bat (None, 14, 14, 64)   192         block5_conv2_Branch_2_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_4_bn (Bat (None, 14, 14, 256)  768         block5_conv2_Branch_3_4_conv[0][0
    __________________________________________________________________________________________________
    block5_conv2_Branch_0_act (Acti (None, 14, 14, 128)  0           block5_conv2_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv2_Branch_1_1_act (Ac (None, 14, 14, 64)   0           block5_conv2_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_2_2_act (Ac (None, 14, 14, 64)   0           block5_conv2_Branch_2_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Branch_3_4_act (Ac (None, 14, 14, 256)  0           block5_conv2_Branch_3_4_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv2_Concatenated (Conc (None, 14, 14, 512)  0           block5_conv2_Branch_0_act[0][0]  
                                                                     block5_conv2_Branch_1_1_act[0][0]
                                                                     block5_conv2_Branch_2_2_act[0][0]
                                                                     block5_conv2_Branch_3_4_act[0][0]
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_conv (Con (None, 14, 14, 16)   8192        block5_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_bn (Batch (None, 14, 14, 16)   48          block5_conv3_Branch_3_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_act (Acti (None, 14, 14, 16)   0           block5_conv3_Branch_3_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_1_conv (C (None, 14, 14, 32)   1536        block5_conv3_Branch_3_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_1_bn (Bat (None, 14, 14, 32)   96          block5_conv3_Branch_3_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_1_act (Ac (None, 14, 14, 32)   0           block5_conv3_Branch_3_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_conv (Con (None, 14, 14, 16)   8192        block5_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_2_conv (C (None, 14, 14, 64)   6144        block5_conv3_Branch_3_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_bn (Batch (None, 14, 14, 16)   48          block5_conv3_Branch_2_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_2_bn (Bat (None, 14, 14, 64)   192         block5_conv3_Branch_3_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_act (Acti (None, 14, 14, 16)   0           block5_conv3_Branch_2_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_2_act (Ac (None, 14, 14, 64)   0           block5_conv3_Branch_3_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_1_conv (C (None, 14, 14, 32)   1536        block5_conv3_Branch_2_act[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_3_conv (C (None, 14, 14, 128)  24576       block5_conv3_Branch_3_2_act[0][0]
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_1_bn (Bat (None, 14, 14, 32)   96          block5_conv3_Branch_2_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_3_bn (Bat (None, 14, 14, 128)  384         block5_conv3_Branch_3_3_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_1 (MaxPooli (None, 14, 14, 512)  0           block5_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_1_act (Ac (None, 14, 14, 32)   0           block5_conv3_Branch_2_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_3_act (Ac (None, 14, 14, 128)  0           block5_conv3_Branch_3_3_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_0_conv (Con (None, 14, 14, 128)  65536       block5_conv2_Concatenated[0][0]  
    __________________________________________________________________________________________________
    block5_conv3_Branch_1_1_conv (C (None, 14, 14, 64)   32768       block5_conv3_Branch_1[0][0]      
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_2_conv (C (None, 14, 14, 64)   6144        block5_conv3_Branch_2_1_act[0][0]
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_4_conv (C (None, 14, 14, 256)  98304       block5_conv3_Branch_3_3_act[0][0]
    __________________________________________________________________________________________________
    block5_conv3_Branch_0_bn (Batch (None, 14, 14, 128)  384         block5_conv3_Branch_0_conv[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_1_1_bn (Bat (None, 14, 14, 64)   192         block5_conv3_Branch_1_1_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_2_bn (Bat (None, 14, 14, 64)   192         block5_conv3_Branch_2_2_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_4_bn (Bat (None, 14, 14, 256)  768         block5_conv3_Branch_3_4_conv[0][0
    __________________________________________________________________________________________________
    block5_conv3_Branch_0_act (Acti (None, 14, 14, 128)  0           block5_conv3_Branch_0_bn[0][0]   
    __________________________________________________________________________________________________
    block5_conv3_Branch_1_1_act (Ac (None, 14, 14, 64)   0           block5_conv3_Branch_1_1_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_2_2_act (Ac (None, 14, 14, 64)   0           block5_conv3_Branch_2_2_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Branch_3_4_act (Ac (None, 14, 14, 256)  0           block5_conv3_Branch_3_4_bn[0][0] 
    __________________________________________________________________________________________________
    block5_conv3_Concatenated (Conc (None, 14, 14, 512)  0           block5_conv3_Branch_0_act[0][0]  
                                                                     block5_conv3_Branch_1_1_act[0][0]
                                                                     block5_conv3_Branch_2_2_act[0][0]
                                                                     block5_conv3_Branch_3_4_act[0][0]
    __________________________________________________________________________________________________
    block5_pool (MaxPooling2D)      (None, 7, 7, 512)    0           block5_conv3_Concatenated[0][0]  
    __________________________________________________________________________________________________
    global_max_pooling2d_1 (GlobalM (None, 512)          0           block5_pool[0][0]                
    ==================================================================================================
    Total params: 7,435,216
    Trainable params: 7,424,368
    Non-trainable params: 10,848
    __________________________________________________________________________________________________
    None
    """


# testBlockInceptionV1()
# testBlockInceptionV3()
testInceptionVGG16()
