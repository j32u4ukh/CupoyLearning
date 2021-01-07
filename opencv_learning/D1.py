import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning"""

path = "data/image/lena.png"

# 以彩色圖片的方式載入
img = cv2.imread(path)

# 以灰階圖片的方式載入
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

showImages(bgr=img, gray=gray)
