import cv2
import numpy as np
from keras.layers.convolutional import Conv2D
from keras.models import Sequential

from utils.opencv import showImages

"""Working directory: CupoyLearning
用實際的影像，嘗試自己搭建一個 1 X 1 和 3 X 3 的模型
看通過 1 X 1 和 3 X 3 卷積層後會有甚麼變化?
大家可以自己嘗試著搭建不同層數後，觀察圖形特徵的變化 
"""

path = "data/image/yolo_dog.jpg"

# 以彩色圖片的方式載入
img = cv2.imread(path)
# showImages(img=img)

"""
Sequential 是一個多層模型
透過 add() 函式將一層一層 layer 加上去
data_format='channels_last' 尺寸为 (batch, rows, cols, channels)
搭建一個 3 個 1*1 的 filters
"""

model = Sequential()
model.add(Conv2D(3, (1, 1),
                 padding="same",
                 data_format='channels_last',
                 activation='relu',
                 input_shape=img.shape))
# 作業: 接續搭建一個 4 個 3*3 的 filters
model.add(Conv2D(4, (3, 3),
                 padding="same",
                 data_format='channels_last',
                 activation='relu',
                 input_shape=img.shape))

print(model.summary())

"""
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 576, 768, 3)       12        
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 576, 768, 4)       112       
=================================================================
Total params: 124
Trainable params: 124
Non-trainable params: 0
_________________________________________________________________
None
"""

model1 = Sequential()
model1.add(Conv2D(3, (1, 1),
                  padding="same",
                  data_format='channels_last',
                  activation='relu',
                  input_shape=img.shape))
print(model1.summary())

"""
Model: "model1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_7 (Conv2D)            (None, 576, 768, 3)       12        
=================================================================
Total params: 12
Trainable params: 12
Non-trainable params: 0
_________________________________________________________________
None
"""

model3 = Sequential()
model3.add(Conv2D(3, (3, 3),
                  padding="same",
                  data_format='channels_last',
                  activation='relu',
                  input_shape=img.shape))
print(model3.summary())

"""
Model: "model3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 576, 768, 3)       84        
=================================================================
Total params: 84
Trainable params: 84
Non-trainable params: 0
_________________________________________________________________
None
"""

# keras 在讀取檔案實是以 batch 的方式一次讀取多張，
# 但我們這裡只需要判讀一張，
# 所以透過 expand_dims() 函式來多擴張一個維度
batch_img = np.expand_dims(img, axis=0)
print(batch_img.shape)

output1 = model1.predict(batch_img)
# output1.shape = (1, 576, 768, 3)

output3 = model3.predict(batch_img)
# output3.shape = (1, 576, 768, 3)

img1 = np.squeeze(output1, axis=0)
img3 = np.squeeze(output3, axis=0)
showImages(img=img, img1=img1, img3=img3)
