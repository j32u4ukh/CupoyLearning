from keras.layers import Activation, BatchNormalization, Conv2D
from keras.models import Sequential

"""Working directory: CupoyLearning

『目標』: 搭建 Conv2D-BN-Activation 層
"""

input_shape = (32, 32, 3)

model = Sequential()

# Conv2D-BN-Activation('sigmoid')
# BatchNormalization主要參數：
# momentum: Momentum for the moving mean and the moving variance.
# epsilon: Small float added to variance to avoid dividing by zero.

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
model.add(Activation('sigmoid'))

# Conv2D-BN-Activation('relu')
model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
model.add(Activation('relu'))

print(model.summary())
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
activation_2 (Activation)    (None, 32, 32, 32)        0         
=================================================================
Total params: 10,400
Trainable params: 10,272
Non-trainable params: 128
_________________________________________________________________
None
"""
