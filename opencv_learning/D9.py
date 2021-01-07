import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning

取得 SIFT 特徵

* 轉成灰階圖片
* 需要額外安裝 OpenCV 相關套件
"""

path = "data/image/nms.jpg"
img = cv2.imread(path)

# 轉為灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 建立 SIFT 物件
"""
如果要透過 OpenCV 使用 SIFT 的話必須要額外安裝擴充的函式庫

為了避免版本問題，我們會指定安裝版本

pip install opencv-contrib-python==3.4.2.16

# 處理權限問題: ERROR: Could not install packages due to an EnvironmentError: [WinError 5] 存取被拒。
pip install --user opencv-contrib-python==3.4.2.16
"""
sift = cv2.xfeatures2d.SIFT_create()

# 取得 SIFT 關鍵點位置
# keypoints = sift.detect(gray, None)
keypoints, features = sift.detectAndCompute(gray, None)
# type(keypoints): list
# #keypoints = 1098
# type(keypoints[0]): cv2.KeyPoint

kp0 = keypoints[0]

# 畫圖 + 顯示圖片
img_show = cv2.drawKeypoints(gray, keypoints, img.copy())
showImages(gray=gray, img_show=img_show)
