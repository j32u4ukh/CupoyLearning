from keras.layers import Activation
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Add
from keras.models import Input, Model

"""Working directory: CupoyLearning

嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Inverted Residual Block.

depthwise's filter shape 爲 (3,3), padding = same
不需要給 alpha, depth multiplier 參數，expansion 因子爲 6
"""


def layerInvertedResidual(input_tensor, expansion):
    """

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

    return added


input_tensor = Input((64, 64, 3))
output = layerInvertedResidual(input_tensor, expansion=6)
model = Model(inputs=input_tensor, outputs=output)
print(model.summary())
"""
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
"""
