import random
import string
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha
from keras import backend as K
from keras.layers import Input, Reshape, Dropout, Dense, Lambda, GRU, Convolution2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from utils.dl import VGG16

"""Working directory: CupoyLearning

搭建一個能識別驗證碼的模型(CNN + CTC)
"""

warnings.simplefilter(action='ignore', category=FutureWarning)
K.tensorflow_backend._get_available_gpus()


# CTC Loss需要四個資訊，分別是
# Label
# 預測
# CNN OUTPUT寬度
# 預測影像所包含文字長度
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 設計generator產生training data
# 產生包含要給loss的資訊
# X=輸入影像
# np.ones(batch_size)*int(conv_shape[2])=CNN輸出feature Map寬度
# np.ones(batch_size)*n_len=字串長度(可浮動)
def gen(batch_size=128):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str))
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size) * int(conv_shape[2]), np.ones(batch_size) * n_len], np.ones(batch_size)


# 驗證碼包含0-10數字以及26個英文字母
characters = string.digits + string.ascii_uppercase
print(characters)

# 設定產生圖片尺寸，以及總類別，n_class之所以要加一是為了留一個位置給Blank
width, height, n_len, n_class = 170, 80, 4, len(characters) + 1

# 設定產生驗證碼的 generator
generator = ImageCaptcha(width=width, height=height)

# 我們先練習固定長度4個字的驗證碼
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

plt.imshow(img)
plt.title(random_str)

rnn_size = 128
input_shape = (height, width, 3)
input_tensor = Input(input_shape)
x = input_tensor

'''自己設計CNN層'''
vgg16 = VGG16(include_top=False, input_shape=input_shape, pooling='max')
middle_model = Model(inputs=vgg16.layers[0].input, outputs=vgg16.layers[10].output)
x = middle_model(x)

# 記錄輸出CNN尺寸，loss部分需要這個資訊
# conv_shape=(Batch_size,輸出高度,輸出寬度,輸出深度)
conv_shape = x.get_shape()

# 從(Batch_size,輸出高度,輸出寬度,輸出深度)變成(Batch_size,輸出寬度,輸出深度*輸出高度)，以符合ctc loss需求
x = Reshape(target_shape=(int(conv_shape[2]), int(conv_shape[1] * conv_shape[3])))(x)

x = Dense('自己設計', activation='relu')(x)

x = Dropout(0.25)(x)
x = Dense(n_class, activation='softmax')(x)

# 包裝用來預測的model
base_model = Model(input=input_tensor, output=x)

# 設定要給CTC Loss的資訊
labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])

# 這裡的model是用來計算loss
model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])

# 之所以要lambda y_true, y_pred: y_pred是因為我們的loss已經包在網路裡，會output:y_true, y_pred，而我們只需要y_pred
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='SGD')

next_ge = gen(batch_size=1)
test_ge = next(next_ge)
plt.imshow(test_ge[0][0][0])
print('Label: ', test_ge[0][1])
print('CNN輸出寬度: ', test_ge[0][2])
print('字串長度(可浮動): ', test_ge[0][3])

model.fit_generator(gen(32), steps_per_epoch=300, epochs=60)

"""預測"""

characters2 = characters + ' '
[X_test, y_test, _, _], _ = next(gen(1))
y_pred = base_model.predict(X_test)
# 用ctc_decode得到解答，自己寫可以參考下方
out = K.get_value(K.ctc_decode(y_pred,
                               input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])
out = ''.join([characters[x] for x in out[0]])
y_true = ''.join([characters[x] for x in y_test[0]])

plt.imshow(X_test[0])
plt.title('pred:' + str(out) + '\ntrue: ' + str(y_true))

argmax = np.argmax(y_pred, axis=2)[0]

"""自己寫 decode CTC"""

# 其中0代表預測為空格，如果預測相同字符之間沒有空格要移除
word = ''
n = ''
for single_result in y_pred[0].argmax(1):
    if single_result != 36:
        if n != single_result:
            word += characters[single_result]
    n = single_result

"""額外參考：加入RNN的神經網路"""

rnn_size = 128

input_tensor = Input((height, width, 3))
x = input_tensor

for i in range(4):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    
    if i < 3:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = MaxPooling2D(pool_size=(2, 1))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[2]), int(conv_shape[1] * conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru1_b')(x)
gru1_merged = keras.layers.add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru2_b')(gru1_merged)
x = keras.layers.Concatenate()([gru_2, gru_2b])

x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)

base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='Adam')
print(model.summary())
