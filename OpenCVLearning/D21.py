from keras.layers import Input
import warnings

import keras
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.dl import batchNormalizationConv2D
from utils.opencv import StandardScaler
from utils.opencv import scalePadding

"""Working directory: CupoyLearning

『本次練習內容』
使用 Keras 做 Transfer Learning (以 Xception 為 backbone)

『本次練習目的』
了解如何使用 Transfer Learning
了解 Transfer Learning 的優點，可以觀察模型收斂速度
"""

warnings.simplefilter(action='ignore', category=FutureWarning)
K.tensorflow_backend._get_available_gpus()


def getXceptionModel(input_shape, n_class=10):
    input_tensor = Input(shape=input_shape)

    model = keras.applications.xception.Xception(include_top=False,
                                                 weights='imagenet',
                                                 input_tensor=input_tensor,
                                                 input_shape=input_shape,
                                                 classes=n_class)

    """
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, 71, 71, 3)    0                                            
    __________________________________________________________________________________________________
    block1_conv1 (Conv2D)           (None, 35, 35, 32)   864         input_2[0][0]                    
    __________________________________________________________________________________________________
    block1_conv1_bn (BatchNormaliza (None, 35, 35, 32)   128         block1_conv1[0][0]               
    __________________________________________________________________________________________________
    block1_conv1_act (Activation)   (None, 35, 35, 32)   0           block1_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block1_conv2 (Conv2D)           (None, 33, 33, 64)   18432       block1_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block1_conv2_bn (BatchNormaliza (None, 33, 33, 64)   256         block1_conv2[0][0]               
    __________________________________________________________________________________________________
    block1_conv2_act (Activation)   (None, 33, 33, 64)   0           block1_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv1 (SeparableConv2 (None, 33, 33, 128)  8768        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_sepconv1_bn (BatchNormal (None, 33, 33, 128)  512         block2_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv2_act (Activation (None, 33, 33, 128)  0           block2_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block2_sepconv2 (SeparableConv2 (None, 33, 33, 128)  17536       block2_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block2_sepconv2_bn (BatchNormal (None, 33, 33, 128)  512         block2_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 17, 17, 128)  8192        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_pool (MaxPooling2D)      (None, 17, 17, 128)  0           block2_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 17, 17, 128)  512         conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 17, 17, 128)  0           block2_pool[0][0]                
                                                                     batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    block3_sepconv1_act (Activation (None, 17, 17, 128)  0           add_1[0][0]                      
    __________________________________________________________________________________________________
    block3_sepconv1 (SeparableConv2 (None, 17, 17, 256)  33920       block3_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv1_bn (BatchNormal (None, 17, 17, 256)  1024        block3_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block3_sepconv2_act (Activation (None, 17, 17, 256)  0           block3_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block3_sepconv2 (SeparableConv2 (None, 17, 17, 256)  67840       block3_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv2_bn (BatchNormal (None, 17, 17, 256)  1024        block3_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 9, 9, 256)    32768       add_1[0][0]                      
    __________________________________________________________________________________________________
    block3_pool (MaxPooling2D)      (None, 9, 9, 256)    0           block3_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 9, 9, 256)    1024        conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 9, 9, 256)    0           block3_pool[0][0]                
                                                                     batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    block4_sepconv1_act (Activation (None, 9, 9, 256)    0           add_2[0][0]                      
    __________________________________________________________________________________________________
    block4_sepconv1 (SeparableConv2 (None, 9, 9, 728)    188672      block4_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv1_bn (BatchNormal (None, 9, 9, 728)    2912        block4_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block4_sepconv2_act (Activation (None, 9, 9, 728)    0           block4_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block4_sepconv2 (SeparableConv2 (None, 9, 9, 728)    536536      block4_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv2_bn (BatchNormal (None, 9, 9, 728)    2912        block4_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 5, 5, 728)    186368      add_2[0][0]                      
    __________________________________________________________________________________________________
    block4_pool (MaxPooling2D)      (None, 5, 5, 728)    0           block4_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 5, 5, 728)    2912        conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 5, 5, 728)    0           block4_pool[0][0]                
                                                                     batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    block5_sepconv1_act (Activation (None, 5, 5, 728)    0           add_3[0][0]                      
    __________________________________________________________________________________________________
    block5_sepconv1 (SeparableConv2 (None, 5, 5, 728)    536536      block5_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv1_bn (BatchNormal (None, 5, 5, 728)    2912        block5_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv2_act (Activation (None, 5, 5, 728)    0           block5_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv2 (SeparableConv2 (None, 5, 5, 728)    536536      block5_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv2_bn (BatchNormal (None, 5, 5, 728)    2912        block5_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv3_act (Activation (None, 5, 5, 728)    0           block5_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv3 (SeparableConv2 (None, 5, 5, 728)    536536      block5_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv3_bn (BatchNormal (None, 5, 5, 728)    2912        block5_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 5, 5, 728)    0           block5_sepconv3_bn[0][0]         
                                                                     add_3[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1_act (Activation (None, 5, 5, 728)    0           add_4[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1 (SeparableConv2 (None, 5, 5, 728)    536536      block6_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv1_bn (BatchNormal (None, 5, 5, 728)    2912        block6_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv2_act (Activation (None, 5, 5, 728)    0           block6_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv2 (SeparableConv2 (None, 5, 5, 728)    536536      block6_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv2_bn (BatchNormal (None, 5, 5, 728)    2912        block6_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv3_act (Activation (None, 5, 5, 728)    0           block6_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv3 (SeparableConv2 (None, 5, 5, 728)    536536      block6_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv3_bn (BatchNormal (None, 5, 5, 728)    2912        block6_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 5, 5, 728)    0           block6_sepconv3_bn[0][0]         
                                                                     add_4[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1_act (Activation (None, 5, 5, 728)    0           add_5[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1 (SeparableConv2 (None, 5, 5, 728)    536536      block7_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv1_bn (BatchNormal (None, 5, 5, 728)    2912        block7_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv2_act (Activation (None, 5, 5, 728)    0           block7_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv2 (SeparableConv2 (None, 5, 5, 728)    536536      block7_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv2_bn (BatchNormal (None, 5, 5, 728)    2912        block7_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv3_act (Activation (None, 5, 5, 728)    0           block7_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv3 (SeparableConv2 (None, 5, 5, 728)    536536      block7_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv3_bn (BatchNormal (None, 5, 5, 728)    2912        block7_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 5, 5, 728)    0           block7_sepconv3_bn[0][0]         
                                                                     add_5[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1_act (Activation (None, 5, 5, 728)    0           add_6[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1 (SeparableConv2 (None, 5, 5, 728)    536536      block8_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv1_bn (BatchNormal (None, 5, 5, 728)    2912        block8_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv2_act (Activation (None, 5, 5, 728)    0           block8_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv2 (SeparableConv2 (None, 5, 5, 728)    536536      block8_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv2_bn (BatchNormal (None, 5, 5, 728)    2912        block8_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv3_act (Activation (None, 5, 5, 728)    0           block8_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv3 (SeparableConv2 (None, 5, 5, 728)    536536      block8_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv3_bn (BatchNormal (None, 5, 5, 728)    2912        block8_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, 5, 5, 728)    0           block8_sepconv3_bn[0][0]         
                                                                     add_6[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1_act (Activation (None, 5, 5, 728)    0           add_7[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1 (SeparableConv2 (None, 5, 5, 728)    536536      block9_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv1_bn (BatchNormal (None, 5, 5, 728)    2912        block9_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv2_act (Activation (None, 5, 5, 728)    0           block9_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv2 (SeparableConv2 (None, 5, 5, 728)    536536      block9_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv2_bn (BatchNormal (None, 5, 5, 728)    2912        block9_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv3_act (Activation (None, 5, 5, 728)    0           block9_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv3 (SeparableConv2 (None, 5, 5, 728)    536536      block9_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv3_bn (BatchNormal (None, 5, 5, 728)    2912        block9_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, 5, 5, 728)    0           block9_sepconv3_bn[0][0]         
                                                                     add_7[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1_act (Activatio (None, 5, 5, 728)    0           add_8[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1 (SeparableConv (None, 5, 5, 728)    536536      block10_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv1_bn (BatchNorma (None, 5, 5, 728)    2912        block10_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv2_act (Activatio (None, 5, 5, 728)    0           block10_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv2 (SeparableConv (None, 5, 5, 728)    536536      block10_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv2_bn (BatchNorma (None, 5, 5, 728)    2912        block10_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv3_act (Activatio (None, 5, 5, 728)    0           block10_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv3 (SeparableConv (None, 5, 5, 728)    536536      block10_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv3_bn (BatchNorma (None, 5, 5, 728)    2912        block10_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, 5, 5, 728)    0           block10_sepconv3_bn[0][0]        
                                                                     add_8[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1_act (Activatio (None, 5, 5, 728)    0           add_9[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1 (SeparableConv (None, 5, 5, 728)    536536      block11_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv1_bn (BatchNorma (None, 5, 5, 728)    2912        block11_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv2_act (Activatio (None, 5, 5, 728)    0           block11_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv2 (SeparableConv (None, 5, 5, 728)    536536      block11_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv2_bn (BatchNorma (None, 5, 5, 728)    2912        block11_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv3_act (Activatio (None, 5, 5, 728)    0           block11_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv3 (SeparableConv (None, 5, 5, 728)    536536      block11_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv3_bn (BatchNorma (None, 5, 5, 728)    2912        block11_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, 5, 5, 728)    0           block11_sepconv3_bn[0][0]        
                                                                     add_9[0][0]                      
    __________________________________________________________________________________________________
    block12_sepconv1_act (Activatio (None, 5, 5, 728)    0           add_10[0][0]                     
    __________________________________________________________________________________________________
    block12_sepconv1 (SeparableConv (None, 5, 5, 728)    536536      block12_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv1_bn (BatchNorma (None, 5, 5, 728)    2912        block12_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv2_act (Activatio (None, 5, 5, 728)    0           block12_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv2 (SeparableConv (None, 5, 5, 728)    536536      block12_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv2_bn (BatchNorma (None, 5, 5, 728)    2912        block12_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv3_act (Activatio (None, 5, 5, 728)    0           block12_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv3 (SeparableConv (None, 5, 5, 728)    536536      block12_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv3_bn (BatchNorma (None, 5, 5, 728)    2912        block12_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, 5, 5, 728)    0           block12_sepconv3_bn[0][0]        
                                                                     add_10[0][0]                     
    __________________________________________________________________________________________________
    block13_sepconv1_act (Activatio (None, 5, 5, 728)    0           add_11[0][0]                     
    __________________________________________________________________________________________________
    block13_sepconv1 (SeparableConv (None, 5, 5, 728)    536536      block13_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv1_bn (BatchNorma (None, 5, 5, 728)    2912        block13_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block13_sepconv2_act (Activatio (None, 5, 5, 728)    0           block13_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block13_sepconv2 (SeparableConv (None, 5, 5, 1024)   752024      block13_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv2_bn (BatchNorma (None, 5, 5, 1024)   4096        block13_sepconv2[0][0]           
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 3, 3, 1024)   745472      add_11[0][0]                     
    __________________________________________________________________________________________________
    block13_pool (MaxPooling2D)     (None, 3, 3, 1024)   0           block13_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 3, 3, 1024)   4096        conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, 3, 3, 1024)   0           block13_pool[0][0]               
                                                                     batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    block14_sepconv1 (SeparableConv (None, 3, 3, 1536)   1582080     add_12[0][0]                     
    __________________________________________________________________________________________________
    block14_sepconv1_bn (BatchNorma (None, 3, 3, 1536)   6144        block14_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv1_act (Activatio (None, 3, 3, 1536)   0           block14_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block14_sepconv2 (SeparableConv (None, 3, 3, 2048)   3159552     block14_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block14_sepconv2_bn (BatchNorma (None, 3, 3, 2048)   8192        block14_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv2_act (Activatio (None, 3, 3, 2048)   0           block14_sepconv2_bn[0][0]        
    ==================================================================================================
    Total params: 20,861,480
    Trainable params: 20,806,952
    Non-trainable params: 54,528
    __________________________________________________________________________________________________
    len(model.layers) = 132
    """

    return model


seed = 7
np.random.seed(seed)

batch_size = 32
n_class = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize Training and Testset
scaler = StandardScaler()
x_train_norm = scaler.fitTransform(x_train)
x_test_norm = scaler.transform(x_test)

x_train_norm = scalePadding(x_train_norm, output_shape=(71, 71))
x_test_norm = scalePadding(x_test_norm, output_shape=(71, 71))

# label to onehot encodeing
y_train_onehot = keras.utils.to_categorical(y_train, n_class)
y_test_onehot = keras.utils.to_categorical(y_test, n_class)

# 劃分'訓練集'與'驗證集'，來對架構與超參數做調整
# 如果數據本身是有序的，需要先 shuffle 再劃分 validation，否則可能會出現驗證集樣本不均勻。
indexs = list(range(len(x_train_norm)))
np.random.shuffle(indexs)
x_train_norm = x_train_norm[indexs]
y_train_onehot = y_train_onehot[indexs]
x_train_norm, x_validation_norm, y_train_onehot, y_validation_onehot = train_test_split(x_train_norm, y_train_onehot,
                                                                                        test_size=0.3,
                                                                                        random_state=seed)

# x_train_norm.shape: (35000, 72, 72, 3), y_train_onehot.shape: (35000, 10)
# x_validation_norm.shape: (15000, 72, 72, 3), y_validation_onehot.shape: (15000, 10)
# x_test_norm.shape: (10000, 72, 72, 3), y_test_onehot.shape: (10000, 10)
print(f"x_train_norm.shape: {x_train_norm.shape}, y_train_onehot.shape: {y_train_onehot.shape}")
print(f"x_validation_norm.shape: {x_validation_norm.shape}, y_validation_onehot.shape: {y_validation_onehot.shape}")
print(f"x_test_norm.shape: {x_test_norm.shape}, y_test_onehot.shape: {y_test_onehot.shape}")


n_class = 10
model = getXceptionModel(input_shape=x_train_norm.shape[1:], n_class=n_class)
x = model.output
x = batchNormalizationConv2D(x=x, filters=64, kernel_size=1)
x = Flatten()(x)
x = Dense(512)(x)
predictions = Dense(activation='softmax', output_dim=n_class)(x)
print(predictions.shape)

model = Model(inputs=model.input, outputs=predictions)
print(len(model.layers))

for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True


# 超過兩個就要選 categorical_crossentrophy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_norm, y_train_onehot,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_validation_norm, y_validation_onehot),
                    shuffle=True,
                    verbose=2)

# 繪制訓練 & 驗證的準確率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 繪制訓練 & 驗證的損失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""模型架構與超參數的調整在驗證集(Validation)獲得良好表現後，才利用測試集衡量表現"""
# y_hat = model.predict(x_test_norm)
scores = model.evaluate(x_test_norm, y_test_onehot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
