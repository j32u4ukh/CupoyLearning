import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning"""

path = "image/lena.png"

# 以彩色圖片的方式載入
img = cv2.imread(path)

# 改變不同的 color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

showImages(bgr=img, hsv=hsv)
