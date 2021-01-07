import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning

實作邊緣檢測 (Sobel Filter)
"""

path = "data/image/lena.png"
img = cv2.imread(path)

# 轉為灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# region 教學
"""
模糊: 透過 Gaussian Filter 實作模糊操作
"""

img_blur = img.copy()

# 重複多次 Gaussian 模糊的操作來加深模糊的程度
img_blur1 = cv2.GaussianBlur(img_blur, (5, 5), 0)
img_blur2 = cv2.GaussianBlur(img_blur1, (5, 5), 0)
img_blur3 = cv2.GaussianBlur(img_blur2, (5, 5), 0)

showImages(img=img, img_blur1=img_blur1, img_blur2=img_blur2, img_blur3=img_blur3)

"""
邊緣檢測: 透過 Sobel Filter 實作邊緣檢測
組合 x-axis, y-axis 的影像合成
"""

# 對 x 方向做 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(gray, cv2.CV_16S, dx=1, dy=0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 對 y 方向做 Sobel 邊緣檢測
img_sobel_y = cv2.Sobel(gray, cv2.CV_16S, dx=0, dy=1, ksize=3)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

# x, y 方向的邊緣檢測後的圖各以一半的全重進行合成
img_sobel_combine = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)

showImages(img=img, img_sobel_x=img_sobel_x, img_sobel_y=img_sobel_y, img_sobel_combine=img_sobel_combine)
# endregion

# region 作業
"""
比較 Sobel 如果在 uint8 的情況下做會 overflow 的狀況
"""

# 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(gray, cv2.CV_16S, dx=1, dy=0, ksize=3)

# 對 x 方向依照比例縮放到所有數值都是非負整數
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
img_sobel_x_uint8 = cv2.Sobel(gray, cv2.CV_8U, dx=1, dy=0, ksize=3)

showImages(gray=gray, img_sobel_x=img_sobel_x, img_sobel_x_uint8=img_sobel_x_uint8)

"""
比較一次與兩次計算偏微分的結果
"""

# 求一次導數取得邊緣檢測結果
img_sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, dx=1, dy=0, ksize=3))

# 求二次導數取得邊緣檢測結果
img_sobel_xx = cv2.convertScaleAbs(cv2.Sobel(img_sobel_x, cv2.CV_16S, dx=1, dy=0, ksize=3))

showImages(gray=gray, img_sobel_x=img_sobel_x, img_sobel_xx=img_sobel_xx)
# endregion
