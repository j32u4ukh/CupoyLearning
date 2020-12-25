import cv2
import imgaug.augmenters.flip as aug
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

"""Working directory: CupoyLearning"""


def loadData(file_name):
    # 讀取 csv 文件
    df = pd.read_csv(f"data/facial-keypoints-detection/{file_name}.csv")

    # 過濾有缺失值的 row
    df = df.dropna()

    # 將圖片像素值讀取為 numpy array 的形態
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ')).values

    # 單獨把圖像 array 抽取出來
    imgs = np.vstack(df['Image'].values) / 255

    # reshape 為 96 x 96
    imgs = imgs.reshape(df.shape[0], 96, 96)

    # 轉換為 float
    imgs = imgs.astype(np.float32)

    # 提取坐標的部分
    points = df[df.columns[:-1]].values

    # 轉換為 float
    points = points.astype(np.float32)

    # normalize 坐標值
    points = points / 96 - 0.5

    return imgs, points


# 回傳定義好的 model 的函數
def buildModel():
    # 定義人臉關鍵點檢測網路
    model = Sequential()

    # 定義神經網路的輸入
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # 最後輸出 30 維的向量，也就是 15 個關鍵點的值
    model.add(Dense(30))
    return model


# 在灰階圖像上畫關鍵點的函數
def plotKeypoints(img, points):
    plt.imshow(img, cmap='gray')
    for i in range(0, 30, 2):
        plt.scatter((points[i] + 0.5) * 96, (points[i + 1] + 0.5) * 96, color='red')


def flip(img, point):
    img = aug.fliplr(img)
    length = len(point)

    idxs = np.arange(0, length, 2)
    point[idxs] = -point[idxs]

    return img, point


imgs_train, points_train = loadData(file_name="training")
print("圖像資料:", imgs_train.shape)
print("關鍵點資料:", points_train.shape)

model = buildModel()

# 載入之前 train 好的權重
model.load_weights("data/facial-keypoints-detection/best_weights.h5")

# 選一張圖片做人臉濾鏡的樣本
sample_img = imgs_train[0]
sample_point = points_train[0]
plotKeypoints(sample_img, sample_point)

# cv2.IMREAD_UNCHANGED 表示要讀取圖像透明度的 channel
sunglasses = cv2.imread('data/image/sunglasses.png', cv2.IMREAD_UNCHANGED)
plt.imshow(sunglasses)

# 預測人臉關鍵點
landmarks = model.predict(sample_img.reshape(-1, 96, 96, 1))

# 將預測的人臉關鍵點的數值範圍由 -0.5 ~ 0.5 轉回 0 ~ 96
landmarks = (landmarks + 0.5) * 96
key_pts = landmarks.reshape(-1, 2)

# 將灰階圖像轉為 BGR
face_img = cv2.cvtColor((sample_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# 以下示範如何用檢測到的關鍵點去定義要增加太陽眼鏡濾鏡的坐標，
# 其中使用的 key_pts index 就是由 plot_keypoints 所畫出來的 index 觀察而來

# 在這裡選右眉毛的最外側 (也就是第 9 index) 做太陽眼鏡的最左邊
sunglass_top_x = int(key_pts[9, 0])

# 在這裡選右眉毛最外側到左眉毛最外側 (也就是第 7、9 index) 做為太陽眼鏡的寬
sunglass_w = int(abs(key_pts[9, 0] - key_pts[7, 0]))

# y 和 h 也是類似的道理了
sunglass_top_y = int((key_pts[9, 1] + key_pts[5, 1]) / 2)
sunglass_h = int(abs(key_pts[8, 1] - key_pts[10, 1]) / 2)

new_sunglasses = cv2.resize(sunglasses, (sunglass_w, sunglass_h), interpolation=cv2.INTER_CUBIC)

# roi 為要覆蓋太陽眼鏡的 BGR 人臉範圍
roi = face_img[sunglass_top_y:sunglass_top_y + sunglass_h, sunglass_top_x:sunglass_top_x + sunglass_w]

# 找出非透明的 pixel
ind = np.argwhere(new_sunglasses[:, :, 3] > 0)

# 把 roi 中每個 channel 非透明的地方替換為太陽眼鏡的 pixel
for i in range(3):
    roi[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]

plt.imshow(face_img)
