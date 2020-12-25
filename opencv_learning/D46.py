from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation
from keras.models import Input, Model

"""Working directory: CupoyLearning

嘗試用 keras 的 DepthwiseConv2D 等 layers 實做 Separable Convolution.


depthwise's filter shape 爲 (3,3), padding = same
pointwise's filters size 爲 128
不需要給 alpha, depth multiplier 參數
"""


def layerSeparableConv(input_tensor):
    """

    :param input_tensor:
    :return:
    """
    x = DepthwiseConv2D(3, padding="same", input_shape=input_tensor.shape)(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)
    return x


input_tensor = Input((64, 64, 3))
output = layerSeparableConv(input_tensor)
model = Model(inputs=input_tensor, outputs=output)
print(model.summary())
"""
Model: "model_1"
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
"""
