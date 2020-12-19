import numpy as np
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Lambda
import warnings
from utils.dl import batchNormalizationConv2D, blockInceptionV1, blockInceptionV3

warnings.simplefilter(action='ignore', category=FutureWarning)

"""Working directory: CupoyLearning"""

"""ResNetV1"""


def residualBlockV1(input_data, kernel_size, filters, stage, block):
    filter1, filter2 = filters
    conv_name_base = f'res_{stage}_{block}_branch'
    bn_name_base = f'bn_{stage}_{block}_branch'

    x = Conv2D(filter1, (1, 1), padding='same', name=conv_name_base + '2a')(input_data)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)

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


"""參考上方 Residual Block 搭建 Inception-ResNet 中的 Inception Block"""


def inceptionResnetBlock(x, scale, block_type, activation='relu'):
    """

    :param x:
    :param scale: scaling factor to scale the residuals (i.e., the output of passing `x` through an
    inception module) before adding them to the shortcut branch. Let `r` be the output
    from the residual branch, the output of this block will be `x + scale * r`.
    (簡單來說就是控制 Residual branch 的比例)
    :param block_type:
    :param activation:
    :return:
    """
    if block_type == 'A':
        branch_0 = batchNormalizationConv2D(x=x, filters=32, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=x, filters=32, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=branch_1, filters=32, kernel_size=3)
        branch_2 = batchNormalizationConv2D(x=x, filters=32, kernel_size=1)
        branch_2 = batchNormalizationConv2D(x=branch_2, filters=48, kernel_size=3)
        branch_2 = batchNormalizationConv2D(x=branch_2, filters=64, kernel_size=3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'B':
        branch_0 = batchNormalizationConv2D(x=x, filters=192, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=x, filters=128, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=branch_1, filters=160, kernel_size=(1, 7))
        branch_1 = batchNormalizationConv2D(x=branch_1, filters=192, kernel_size=(7, 1))
        branches = [branch_0, branch_1]
    elif block_type == 'C':
        branch_0 = batchNormalizationConv2D(x=x, filters=192, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=x, filters=192, kernel_size=1)
        branch_1 = batchNormalizationConv2D(x=branch_1, filters=192, kernel_size=(1, 3))
        branch_1 = batchNormalizationConv2D(x=branch_1, filters=192, kernel_size=(3, 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))
    mixed = Concatenate(axis=3)(branches)

    '''確保輸入跟輸出深度相同'''
    up = batchNormalizationConv2D(x=mixed, filters=K.int_shape(x)[3], kernel_size=1, activation=None)

    '''導入殘差結構，並給予權重'''
    # inputs[0]: x, inputs[1]: up
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale}, )([x, up])

    if activation is not None:
        x = Activation(activation)(x)
    return x


def inceptionResnetVGG16(include_top=True, input_shape=(224, 224, 1), pooling='max', classes=1000):
    img_input = Input(shape=input_shape)

    x = batchNormalizationConv2D(img_input, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
    x = batchNormalizationConv2D(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = inceptionResnetBlock(x, 0.1, 'A', activation='relu')
    x = batchNormalizationConv2D(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv1')
    x = batchNormalizationConv2D(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_1')
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_2')
    x = blockInceptionV1(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = inceptionResnetBlock(x, 0.1, 'B', activation='relu')
    x = batchNormalizationConv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
    x = inceptionResnetBlock(x, 0.1, 'C', activation='relu')
    x = batchNormalizationConv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
    x = batchNormalizationConv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')

    # Block 4 MaxPooling2D: (?, 14, 14, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    print("Block 4 MaxPooling2D:", x.shape)

    # Block 5
    # 為什麼要加InceptionV3_block 原因?
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)), filter_size=3, channel_axis=3,
                         name='Block_4')
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)), filter_size=3, channel_axis=3,
                         name='Block_5')
    x = blockInceptionV3(x, ((128,), (64,), (16, 32, 64), (16, 32, 64, 128, 256)), filter_size=3, channel_axis=3,
                         name='Block_6')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        # 可以提醒學員為什麼要加avg或是max
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    return model


"""Test"""


def testResidualBlockV1():
    input_data = Input(shape=(224, 224, 32))
    kernel_size = (3, 3)
    filters = (16, 32)
    stage = "test"
    block = 0
    result = residualBlockV1(input_data, kernel_size, filters, stage, block)
    print("result.shape:", result.shape)


def testResidualBlockV2():
    input_data = Input(shape=(224, 224, 32))
    kernel_size = (3, 3)
    filters = (16, 32)
    stage = "test"
    block = 0
    result = residualBlockV2(input_data, kernel_size, filters, stage, block)
    print("result.shape:", result.shape)


def testResidualZipBlockV2():
    input_data = Input(shape=(224, 224, 32))
    kernel_size = (3, 3)
    reduce = 8
    ouput_size = 32
    stage = "test"
    block = 0
    result = residualZipBlockV2(input_data, kernel_size, stage, block, reduce, ouput_size)
    print("result.shape:", result.shape)


def testInceptionResnetBlock():
    img_input = Input(shape=(224, 224, 32))
    a = inceptionResnetBlock(img_input, 0.1, 'A', activation='relu')
    print(a)

    b = inceptionResnetBlock(img_input, 0.1, 'B', activation='relu')
    print(b)

    c = inceptionResnetBlock(img_input, 0.1, 'C', activation='relu')
    print(c)


def testInceptionResnetVGG16():
    model = inceptionResnetVGG16(include_top=False)
    print(model.summary())

    """
    Model: "vgg16"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_18 (InputLayer)           (None, 224, 224, 1)  0                                            
    __________________________________________________________________________________________________
    block1_conv1_conv (Conv2D)      (None, 224, 224, 64) 576         input_18[0][0]                   
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
    conv2d_67 (Conv2D)              (None, 112, 112, 32) 2048        block1_pool[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 112, 112, 32) 96          conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, 112, 112, 32) 0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 112, 112, 32) 2048        block1_pool[0][0]                
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 112, 112, 48) 13824       activation_80[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 112, 112, 32) 96          conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 112, 112, 48) 144         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, 112, 112, 32) 0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, 112, 112, 48) 0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 112, 112, 32) 2048        block1_pool[0][0]                
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 112, 112, 32) 9216        activation_78[0][0]              
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, 112, 112, 64) 27648       activation_81[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 112, 112, 32) 96          conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 112, 112, 32) 96          conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 112, 112, 64) 192         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, 112, 112, 32) 0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, 112, 112, 32) 0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, 112, 112, 64) 0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    concatenate_11 (Concatenate)    (None, 112, 112, 128 0           activation_77[0][0]              
                                                                     activation_79[0][0]              
                                                                     activation_82[0][0]              
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, 112, 112, 64) 8192        concatenate_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 112, 112, 64) 192         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    lambda_11 (Lambda)              (None, 112, 112, 64) 0           block1_pool[0][0]                
                                                                     batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, 112, 112, 64) 0           lambda_11[0][0]                  
    __________________________________________________________________________________________________
    block2_conv1_conv (Conv2D)      (None, 112, 112, 128 73728       activation_83[0][0]              
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
    Block_1_Branch_1_conv (Conv2D)  (None, 56, 56, 96)   12288       block2_pool[0][0]                
    __________________________________________________________________________________________________
    Block_1_Branch_2_conv (Conv2D)  (None, 56, 56, 16)   2048        block2_pool[0][0]                
    __________________________________________________________________________________________________
    Block_1_Branch_1_bn (BatchNorma (None, 56, 56, 96)   288         Block_1_Branch_1_conv[0][0]      
    __________________________________________________________________________________________________
    Block_1_Branch_2_bn (BatchNorma (None, 56, 56, 16)   48          Block_1_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_1_Branch_1_act (Activatio (None, 56, 56, 96)   0           Block_1_Branch_1_bn[0][0]        
    __________________________________________________________________________________________________
    Block_1_Branch_2_act (Activatio (None, 56, 56, 16)   0           Block_1_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_1_Branch_3 (MaxPooling2D) (None, 56, 56, 128)  0           block2_pool[0][0]                
    __________________________________________________________________________________________________
    Block_1_Branch_0_conv (Conv2D)  (None, 56, 56, 64)   8192        block2_pool[0][0]                
    __________________________________________________________________________________________________
    Block_1_Branch_1_1_conv (Conv2D (None, 56, 56, 128)  110592      Block_1_Branch_1_act[0][0]       
    __________________________________________________________________________________________________
    Block_1_Branch_2_1_conv (Conv2D (None, 56, 56, 32)   12800       Block_1_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_1_Branch_3_1_conv (Conv2D (None, 56, 56, 32)   4096        Block_1_Branch_3[0][0]           
    __________________________________________________________________________________________________
    Block_1_Branch_0_bn (BatchNorma (None, 56, 56, 64)   192         Block_1_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_1_Branch_1_1_bn (BatchNor (None, 56, 56, 128)  384         Block_1_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_1_Branch_2_1_bn (BatchNor (None, 56, 56, 32)   96          Block_1_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_1_Branch_3_1_bn (BatchNor (None, 56, 56, 32)   96          Block_1_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_1_Branch_0_act (Activatio (None, 56, 56, 64)   0           Block_1_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_1_Branch_1_1_act (Activat (None, 56, 56, 128)  0           Block_1_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_1_Branch_2_1_act (Activat (None, 56, 56, 32)   0           Block_1_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_1_Branch_3_1_act (Activat (None, 56, 56, 32)   0           Block_1_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_1_Concatenated (Concatena (None, 56, 56, 256)  0           Block_1_Branch_0_act[0][0]       
                                                                     Block_1_Branch_1_1_act[0][0]     
                                                                     Block_1_Branch_2_1_act[0][0]     
                                                                     Block_1_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_2_Branch_1_conv (Conv2D)  (None, 56, 56, 96)   24576       Block_1_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_2_conv (Conv2D)  (None, 56, 56, 16)   4096        Block_1_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_1_bn (BatchNorma (None, 56, 56, 96)   288         Block_2_Branch_1_conv[0][0]      
    __________________________________________________________________________________________________
    Block_2_Branch_2_bn (BatchNorma (None, 56, 56, 16)   48          Block_2_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_2_Branch_1_act (Activatio (None, 56, 56, 96)   0           Block_2_Branch_1_bn[0][0]        
    __________________________________________________________________________________________________
    Block_2_Branch_2_act (Activatio (None, 56, 56, 16)   0           Block_2_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_2_Branch_3 (MaxPooling2D) (None, 56, 56, 256)  0           Block_1_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_0_conv (Conv2D)  (None, 56, 56, 64)   16384       Block_1_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_1_1_conv (Conv2D (None, 56, 56, 128)  110592      Block_2_Branch_1_act[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_2_1_conv (Conv2D (None, 56, 56, 32)   12800       Block_2_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_2_Branch_3_1_conv (Conv2D (None, 56, 56, 32)   8192        Block_2_Branch_3[0][0]           
    __________________________________________________________________________________________________
    Block_2_Branch_0_bn (BatchNorma (None, 56, 56, 64)   192         Block_2_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_2_Branch_1_1_bn (BatchNor (None, 56, 56, 128)  384         Block_2_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_2_Branch_2_1_bn (BatchNor (None, 56, 56, 32)   96          Block_2_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_2_Branch_3_1_bn (BatchNor (None, 56, 56, 32)   96          Block_2_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_2_Branch_0_act (Activatio (None, 56, 56, 64)   0           Block_2_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_2_Branch_1_1_act (Activat (None, 56, 56, 128)  0           Block_2_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_2_Branch_2_1_act (Activat (None, 56, 56, 32)   0           Block_2_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_2_Branch_3_1_act (Activat (None, 56, 56, 32)   0           Block_2_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_2_Concatenated (Concatena (None, 56, 56, 256)  0           Block_2_Branch_0_act[0][0]       
                                                                     Block_2_Branch_1_1_act[0][0]     
                                                                     Block_2_Branch_2_1_act[0][0]     
                                                                     Block_2_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_3_Branch_1_conv (Conv2D)  (None, 56, 56, 96)   24576       Block_2_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_2_conv (Conv2D)  (None, 56, 56, 16)   4096        Block_2_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_1_bn (BatchNorma (None, 56, 56, 96)   288         Block_3_Branch_1_conv[0][0]      
    __________________________________________________________________________________________________
    Block_3_Branch_2_bn (BatchNorma (None, 56, 56, 16)   48          Block_3_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_3_Branch_1_act (Activatio (None, 56, 56, 96)   0           Block_3_Branch_1_bn[0][0]        
    __________________________________________________________________________________________________
    Block_3_Branch_2_act (Activatio (None, 56, 56, 16)   0           Block_3_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_3_Branch_3 (MaxPooling2D) (None, 56, 56, 256)  0           Block_2_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_0_conv (Conv2D)  (None, 56, 56, 64)   16384       Block_2_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_1_1_conv (Conv2D (None, 56, 56, 128)  110592      Block_3_Branch_1_act[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_2_1_conv (Conv2D (None, 56, 56, 32)   12800       Block_3_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_3_Branch_3_1_conv (Conv2D (None, 56, 56, 32)   8192        Block_3_Branch_3[0][0]           
    __________________________________________________________________________________________________
    Block_3_Branch_0_bn (BatchNorma (None, 56, 56, 64)   192         Block_3_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_3_Branch_1_1_bn (BatchNor (None, 56, 56, 128)  384         Block_3_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_3_Branch_2_1_bn (BatchNor (None, 56, 56, 32)   96          Block_3_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_3_Branch_3_1_bn (BatchNor (None, 56, 56, 32)   96          Block_3_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_3_Branch_0_act (Activatio (None, 56, 56, 64)   0           Block_3_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_3_Branch_1_1_act (Activat (None, 56, 56, 128)  0           Block_3_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_3_Branch_2_1_act (Activat (None, 56, 56, 32)   0           Block_3_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_3_Branch_3_1_act (Activat (None, 56, 56, 32)   0           Block_3_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_3_Concatenated (Concatena (None, 56, 56, 256)  0           Block_3_Branch_0_act[0][0]       
                                                                     Block_3_Branch_1_1_act[0][0]     
                                                                     Block_3_Branch_2_1_act[0][0]     
                                                                     Block_3_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    block3_pool (MaxPooling2D)      (None, 28, 28, 256)  0           Block_3_Concatenated[0][0]       
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, 28, 28, 128)  32768       block3_pool[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 28, 28, 128)  384         conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, 28, 28, 128)  0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, 28, 28, 160)  143360      activation_85[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 28, 28, 160)  480         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, 28, 28, 160)  0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, 28, 28, 192)  49152       block3_pool[0][0]                
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, 28, 28, 192)  215040      activation_86[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 28, 28, 192)  576         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, 28, 28, 192)  576         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, 28, 28, 192)  0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, 28, 28, 192)  0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    concatenate_12 (Concatenate)    (None, 28, 28, 384)  0           activation_84[0][0]              
                                                                     activation_87[0][0]              
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, 28, 28, 256)  98304       concatenate_12[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, 28, 28, 256)  768         conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    lambda_12 (Lambda)              (None, 28, 28, 256)  0           block3_pool[0][0]                
                                                                     batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, 28, 28, 256)  0           lambda_12[0][0]                  
    __________________________________________________________________________________________________
    block4_conv1_conv (Conv2D)      (None, 28, 28, 512)  1179648     activation_88[0][0]              
    __________________________________________________________________________________________________
    block4_conv1_bn (BatchNormaliza (None, 28, 28, 512)  1536        block4_conv1_conv[0][0]          
    __________________________________________________________________________________________________
    block4_conv1_act (Activation)   (None, 28, 28, 512)  0           block4_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, 28, 28, 192)  98304       block4_conv1_act[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, 28, 28, 192)  576         conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, 28, 28, 192)  0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, 28, 28, 192)  110592      activation_90[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, 28, 28, 192)  576         conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, 28, 28, 192)  0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, 28, 28, 192)  98304       block4_conv1_act[0][0]           
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, 28, 28, 192)  110592      activation_91[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, 28, 28, 192)  576         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, 28, 28, 192)  576         conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, 28, 28, 192)  0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, 28, 28, 192)  0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    concatenate_13 (Concatenate)    (None, 28, 28, 384)  0           activation_89[0][0]              
                                                                     activation_92[0][0]              
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, 28, 28, 512)  196608      concatenate_13[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, 28, 28, 512)  1536        conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    lambda_13 (Lambda)              (None, 28, 28, 512)  0           block4_conv1_act[0][0]           
                                                                     batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, 28, 28, 512)  0           lambda_13[0][0]                  
    __________________________________________________________________________________________________
    block4_conv2_conv (Conv2D)      (None, 28, 28, 512)  2359296     activation_93[0][0]              
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
    Block_4_Branch_3_conv (Conv2D)  (None, 14, 14, 16)   8192        block4_pool[0][0]                
    __________________________________________________________________________________________________
    Block_4_Branch_3_bn (BatchNorma (None, 14, 14, 16)   48          Block_4_Branch_3_conv[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_3_act (Activatio (None, 14, 14, 16)   0           Block_4_Branch_3_bn[0][0]        
    __________________________________________________________________________________________________
    Block_4_Branch_3_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_4_Branch_3_act[0][0]       
    __________________________________________________________________________________________________
    Block_4_Branch_3_1_bn (BatchNor (None, 14, 14, 32)   96          Block_4_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_3_1_act (Activat (None, 14, 14, 32)   0           Block_4_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_2_conv (Conv2D)  (None, 14, 14, 16)   8192        block4_pool[0][0]                
    __________________________________________________________________________________________________
    Block_4_Branch_3_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_4_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_4_Branch_2_bn (BatchNorma (None, 14, 14, 16)   48          Block_4_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_3_2_bn (BatchNor (None, 14, 14, 64)   192         Block_4_Branch_3_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_2_act (Activatio (None, 14, 14, 16)   0           Block_4_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_4_Branch_3_2_act (Activat (None, 14, 14, 64)   0           Block_4_Branch_3_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_2_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_4_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_4_Branch_3_3_conv (Conv2D (None, 14, 14, 128)  24576       Block_4_Branch_3_2_act[0][0]     
    __________________________________________________________________________________________________
    Block_4_Branch_2_1_bn (BatchNor (None, 14, 14, 32)   96          Block_4_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_3_3_bn (BatchNor (None, 14, 14, 128)  384         Block_4_Branch_3_3_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_1 (MaxPooling2D) (None, 14, 14, 512)  0           block4_pool[0][0]                
    __________________________________________________________________________________________________
    Block_4_Branch_2_1_act (Activat (None, 14, 14, 32)   0           Block_4_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_3_3_act (Activat (None, 14, 14, 128)  0           Block_4_Branch_3_3_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_0_conv (Conv2D)  (None, 14, 14, 128)  65536       block4_pool[0][0]                
    __________________________________________________________________________________________________
    Block_4_Branch_1_1_conv (Conv2D (None, 14, 14, 64)   32768       Block_4_Branch_1[0][0]           
    __________________________________________________________________________________________________
    Block_4_Branch_2_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_4_Branch_2_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_4_Branch_3_4_conv (Conv2D (None, 14, 14, 256)  98304       Block_4_Branch_3_3_act[0][0]     
    __________________________________________________________________________________________________
    Block_4_Branch_0_bn (BatchNorma (None, 14, 14, 128)  384         Block_4_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_1_1_bn (BatchNor (None, 14, 14, 64)   192         Block_4_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_2_2_bn (BatchNor (None, 14, 14, 64)   192         Block_4_Branch_2_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_3_4_bn (BatchNor (None, 14, 14, 256)  768         Block_4_Branch_3_4_conv[0][0]    
    __________________________________________________________________________________________________
    Block_4_Branch_0_act (Activatio (None, 14, 14, 128)  0           Block_4_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_4_Branch_1_1_act (Activat (None, 14, 14, 64)   0           Block_4_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_2_2_act (Activat (None, 14, 14, 64)   0           Block_4_Branch_2_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Branch_3_4_act (Activat (None, 14, 14, 256)  0           Block_4_Branch_3_4_bn[0][0]      
    __________________________________________________________________________________________________
    Block_4_Concatenated (Concatena (None, 14, 14, 512)  0           Block_4_Branch_0_act[0][0]       
                                                                     Block_4_Branch_1_1_act[0][0]     
                                                                     Block_4_Branch_2_2_act[0][0]     
                                                                     Block_4_Branch_3_4_act[0][0]     
    __________________________________________________________________________________________________
    Block_5_Branch_3_conv (Conv2D)  (None, 14, 14, 16)   8192        Block_4_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_3_bn (BatchNorma (None, 14, 14, 16)   48          Block_5_Branch_3_conv[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_3_act (Activatio (None, 14, 14, 16)   0           Block_5_Branch_3_bn[0][0]        
    __________________________________________________________________________________________________
    Block_5_Branch_3_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_5_Branch_3_act[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_3_1_bn (BatchNor (None, 14, 14, 32)   96          Block_5_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_3_1_act (Activat (None, 14, 14, 32)   0           Block_5_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_2_conv (Conv2D)  (None, 14, 14, 16)   8192        Block_4_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_3_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_5_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_5_Branch_2_bn (BatchNorma (None, 14, 14, 16)   48          Block_5_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_3_2_bn (BatchNor (None, 14, 14, 64)   192         Block_5_Branch_3_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_2_act (Activatio (None, 14, 14, 16)   0           Block_5_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_5_Branch_3_2_act (Activat (None, 14, 14, 64)   0           Block_5_Branch_3_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_2_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_5_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_3_3_conv (Conv2D (None, 14, 14, 128)  24576       Block_5_Branch_3_2_act[0][0]     
    __________________________________________________________________________________________________
    Block_5_Branch_2_1_bn (BatchNor (None, 14, 14, 32)   96          Block_5_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_3_3_bn (BatchNor (None, 14, 14, 128)  384         Block_5_Branch_3_3_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_1 (MaxPooling2D) (None, 14, 14, 512)  0           Block_4_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_2_1_act (Activat (None, 14, 14, 32)   0           Block_5_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_3_3_act (Activat (None, 14, 14, 128)  0           Block_5_Branch_3_3_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_0_conv (Conv2D)  (None, 14, 14, 128)  65536       Block_4_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_5_Branch_1_1_conv (Conv2D (None, 14, 14, 64)   32768       Block_5_Branch_1[0][0]           
    __________________________________________________________________________________________________
    Block_5_Branch_2_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_5_Branch_2_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_5_Branch_3_4_conv (Conv2D (None, 14, 14, 256)  98304       Block_5_Branch_3_3_act[0][0]     
    __________________________________________________________________________________________________
    Block_5_Branch_0_bn (BatchNorma (None, 14, 14, 128)  384         Block_5_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_1_1_bn (BatchNor (None, 14, 14, 64)   192         Block_5_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_2_2_bn (BatchNor (None, 14, 14, 64)   192         Block_5_Branch_2_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_3_4_bn (BatchNor (None, 14, 14, 256)  768         Block_5_Branch_3_4_conv[0][0]    
    __________________________________________________________________________________________________
    Block_5_Branch_0_act (Activatio (None, 14, 14, 128)  0           Block_5_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_5_Branch_1_1_act (Activat (None, 14, 14, 64)   0           Block_5_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_2_2_act (Activat (None, 14, 14, 64)   0           Block_5_Branch_2_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Branch_3_4_act (Activat (None, 14, 14, 256)  0           Block_5_Branch_3_4_bn[0][0]      
    __________________________________________________________________________________________________
    Block_5_Concatenated (Concatena (None, 14, 14, 512)  0           Block_5_Branch_0_act[0][0]       
                                                                     Block_5_Branch_1_1_act[0][0]     
                                                                     Block_5_Branch_2_2_act[0][0]     
                                                                     Block_5_Branch_3_4_act[0][0]     
    __________________________________________________________________________________________________
    Block_6_Branch_3_conv (Conv2D)  (None, 14, 14, 16)   8192        Block_5_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_3_bn (BatchNorma (None, 14, 14, 16)   48          Block_6_Branch_3_conv[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_3_act (Activatio (None, 14, 14, 16)   0           Block_6_Branch_3_bn[0][0]        
    __________________________________________________________________________________________________
    Block_6_Branch_3_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_6_Branch_3_act[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_3_1_bn (BatchNor (None, 14, 14, 32)   96          Block_6_Branch_3_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_3_1_act (Activat (None, 14, 14, 32)   0           Block_6_Branch_3_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_2_conv (Conv2D)  (None, 14, 14, 16)   8192        Block_5_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_3_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_6_Branch_3_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_6_Branch_2_bn (BatchNorma (None, 14, 14, 16)   48          Block_6_Branch_2_conv[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_3_2_bn (BatchNor (None, 14, 14, 64)   192         Block_6_Branch_3_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_2_act (Activatio (None, 14, 14, 16)   0           Block_6_Branch_2_bn[0][0]        
    __________________________________________________________________________________________________
    Block_6_Branch_3_2_act (Activat (None, 14, 14, 64)   0           Block_6_Branch_3_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_2_1_conv (Conv2D (None, 14, 14, 32)   1536        Block_6_Branch_2_act[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_3_3_conv (Conv2D (None, 14, 14, 128)  24576       Block_6_Branch_3_2_act[0][0]     
    __________________________________________________________________________________________________
    Block_6_Branch_2_1_bn (BatchNor (None, 14, 14, 32)   96          Block_6_Branch_2_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_3_3_bn (BatchNor (None, 14, 14, 128)  384         Block_6_Branch_3_3_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_1 (MaxPooling2D) (None, 14, 14, 512)  0           Block_5_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_2_1_act (Activat (None, 14, 14, 32)   0           Block_6_Branch_2_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_3_3_act (Activat (None, 14, 14, 128)  0           Block_6_Branch_3_3_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_0_conv (Conv2D)  (None, 14, 14, 128)  65536       Block_5_Concatenated[0][0]       
    __________________________________________________________________________________________________
    Block_6_Branch_1_1_conv (Conv2D (None, 14, 14, 64)   32768       Block_6_Branch_1[0][0]           
    __________________________________________________________________________________________________
    Block_6_Branch_2_2_conv (Conv2D (None, 14, 14, 64)   6144        Block_6_Branch_2_1_act[0][0]     
    __________________________________________________________________________________________________
    Block_6_Branch_3_4_conv (Conv2D (None, 14, 14, 256)  98304       Block_6_Branch_3_3_act[0][0]     
    __________________________________________________________________________________________________
    Block_6_Branch_0_bn (BatchNorma (None, 14, 14, 128)  384         Block_6_Branch_0_conv[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_1_1_bn (BatchNor (None, 14, 14, 64)   192         Block_6_Branch_1_1_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_2_2_bn (BatchNor (None, 14, 14, 64)   192         Block_6_Branch_2_2_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_3_4_bn (BatchNor (None, 14, 14, 256)  768         Block_6_Branch_3_4_conv[0][0]    
    __________________________________________________________________________________________________
    Block_6_Branch_0_act (Activatio (None, 14, 14, 128)  0           Block_6_Branch_0_bn[0][0]        
    __________________________________________________________________________________________________
    Block_6_Branch_1_1_act (Activat (None, 14, 14, 64)   0           Block_6_Branch_1_1_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_2_2_act (Activat (None, 14, 14, 64)   0           Block_6_Branch_2_2_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Branch_3_4_act (Activat (None, 14, 14, 256)  0           Block_6_Branch_3_4_bn[0][0]      
    __________________________________________________________________________________________________
    Block_6_Concatenated (Concatena (None, 14, 14, 512)  0           Block_6_Branch_0_act[0][0]       
                                                                     Block_6_Branch_1_1_act[0][0]     
                                                                     Block_6_Branch_2_2_act[0][0]     
                                                                     Block_6_Branch_3_4_act[0][0]     
    __________________________________________________________________________________________________
    block5_pool (MaxPooling2D)      (None, 7, 7, 512)    0           Block_6_Concatenated[0][0]       
    __________________________________________________________________________________________________
    global_max_pooling2d_1 (GlobalM (None, 512)          0           block5_pool[0][0]                
    ==================================================================================================
    Total params: 8,660,800
    Trainable params: 8,644,928
    Non-trainable params: 15,872
    __________________________________________________________________________________________________
    None
    """


# testResidualBlockV1()
# testResidualBlockV2()
# testResidualZipBlockV2()
# testInceptionResnetBlock()
testInceptionResnetVGG16()
