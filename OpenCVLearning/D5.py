import cv2
import numpy as np

from utils.opencv import showImages

"""Working directory: CupoyLearning

Hint: 人物原始邊框座標 (60, 40), (420, 510)

請根據 Lena 圖做以下處理

* 對明亮度做直方圖均衡處理
* 水平鏡像 + 縮放處理 (0.5 倍)
* 畫出人物矩形邊框
"""

path = "image/lena.png"
img = cv2.imread(path)
left_top = [60, 40]
right_bottom = [420, 510]

"""
對明亮度做直方圖均衡處理
"""

hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
hls[..., 1] = cv2.equalizeHist(hls[..., 1])
bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

"""
水平鏡像 + 縮放處理 (0.5 倍)
"""
horizontal_flip = bgr[:, ::-1, :]

height, width, _ = bgr.shape
left_top[1] = width - left_top[1] - 1
right_bottom[1] = width - right_bottom[1] - 1

# 建構 scale matrix
fx = 0.5
fy = 0.5
M = np.array([[fx, 0, 0],
              [0, fy, 0]],
             dtype=np.float32)
small = cv2.warpAffine(horizontal_flip, M, (horizontal_flip.shape[1], horizontal_flip.shape[0]))

"""
畫出人物矩形邊框
"""
# 把左上跟右下轉為矩陣型式
box = np.array((left_top, right_bottom), dtype=np.float32)

# 做矩陣乘法可以使用 `np.dot`, 為了做矩陣乘法, M_scale 需要做轉置之後才能相乘
homo_coor_result = np.dot(M.T, box)
homo_coor_result = homo_coor_result.astype('uint8')

scale_point1 = tuple(homo_coor_result[0])
scale_point2 = tuple(homo_coor_result[1])
print('origin point1={}, origin point2={}'.format(left_top, right_bottom))
print('resize point1={}, resize point2={}'.format(scale_point1, scale_point2))


# 畫圖
cv2.rectangle(small, scale_point1, scale_point2, (0, 0, 255), 3)
showImages(dst=small)

"""
Hint: 矩形
"""

img_rect = img.copy()
cv2.rectangle(
    # 圖片
    img_rect,
    # 左上角
    (60, 40),
    # 右下角
    (420, 510),
    # 顏色
    (0, 0, 255),
    # 線的粗細(若為 -1 則填滿整個矩形)
    3)

showImages(img=img, img_rect=img_rect)

"""
Hint: 線
"""

img_line = img.copy()
cv2.line(img_line, (60, 40), (420, 510), (0, 0, 255), 3)

showImages(img=img, img_line=img_line)

"""
Hint: 文字
"""

img_text = img.copy()
cv2.putText(
    # 圖片
    img_text,
    # 要添加的文字
    '(60, 40)',
    # 文字左下角位置
    (60, 40),
    # 字型
    0,
    # 字體大小
    1,
    # 顏色
    (0, 0, 255),
    # 字體粗細
    2)

showImages(img=img, img_text=img_text)


