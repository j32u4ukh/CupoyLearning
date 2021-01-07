import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning
調整 飽和 / 對比 / 明亮

* 改變到 HSL color space 來調整飽和度
* 對灰階圖實作直方圖均衡
* alpha / beta 調整 對比 / 明亮
"""

path = "data/image/lena.png"

# 以彩色圖片的方式載入
img = cv2.imread(path)

"""
改變飽和度
轉換成 HLS color space, 改變 s channel 的值
"""

# 改變不同的 color space
# HSL: 色相，飽和度，亮度
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

change_percentage = 0.2

# 針對飽和度的值做改變，超過界線 0~1 的都會 bound
# 在 HLS color space 減少飽和度
lower_saturation = hls.astype('float32').copy() / 255.0
lower_saturation[..., -1] = np.clip(lower_saturation[..., -1] - change_percentage, 0.0, 1.0) * 255.0
lower_saturation[..., -1] = np.int8(lower_saturation[..., -1])

# 在 HLS color space 增加飽和度
higher_saturation = hls.astype('float32').copy() / 255.0
higher_saturation[..., -1] = np.clip(higher_saturation[..., -1] + change_percentage, 0.0, 1.0) * 255.0
higher_saturation[..., -1] = np.int8(higher_saturation[..., -1])

# 轉換
lower_saturation = cv2.cvtColor(lower_saturation, cv2.COLOR_HLS2BGR)
higher_saturation = cv2.cvtColor(higher_saturation, cv2.COLOR_HLS2BGR)

# 組合圖片 + 顯示圖片
img_hls_change = np.hstack((img, lower_saturation, higher_saturation))

showImages(hls_change=img_hls_change)

"""
直方圖均衡
"""

# case 1: 把彩圖拆開對每個 channel 個別做直方圖均衡再組合起來
equal_hist = img.copy()
equal_hist[..., 0] = cv2.equalizeHist(equal_hist[..., 0])
equal_hist[..., 1] = cv2.equalizeHist(equal_hist[..., 1])
equal_hist[..., 2] = cv2.equalizeHist(equal_hist[..., 2])
showImages(img=img, equal_hist0=equal_hist)

hls_equal_hist = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HLS)
hls_equal_hist[..., 2] = cv2.equalizeHist(hls_equal_hist[..., 2])
showImages(hls=hls, hls_equal_hist=hls_equal_hist)

gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
gray_equal_hist = cv2.equalizeHist(gray)
showImages(gray=gray, gray_equal_hist=gray_equal_hist)

"""
調整對比 / 明亮
"""

# alpha: 控制對比度 (1.0~3.0)
# beta: 控制明亮度 (0~255)
add_contrast = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
add_lighness = cv2.convertScaleAbs(img, alpha=1.0, beta=50)

# 組合圖片 + 顯示圖片
contrast_light = np.hstack((img, add_contrast, add_lighness))

showImages(contrast_light=contrast_light)
