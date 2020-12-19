import cv2

from utils.opencv import showImages

"""Working directory: CupoyLearning

透過 SIFT 特徵實作 Brute-Force Matching
"""

# 以灰階方式讀入圖片
query = cv2.imread('image/box.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('image/box_in_scene.png', cv2.IMREAD_GRAYSCALE)


# 建立 SIFT 物件
sift = cv2.xfeatures2d_SIFT.create()

# 偵測並計算 SIFT 特徵 (keypoints 關鍵點, descriptor 128 維敘述子)
kp_query, des_query = sift.detectAndCompute(query, None)
kp_target, des_target = sift.detectAndCompute(target, None)

"""
基於 SIFT 特徵的暴力比對

* D.Lowe ratio test
* knn 比對
"""

# 建立 Brute-Force Matching 物件
bf = cv2.BFMatcher(cv2.NORM_L2)

# 以 knn 方式暴力比對特徵
matches = bf.knnMatch(des_query, des_target, k=2)

# 透過 D.Lowe ratio test 排除不適合的配對
candidate = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        candidate.append([m])

# 顯示配對結果
dst = cv2.drawMatchesKnn(query, kp_query, target, kp_target, candidate, None, flags=2)

showImages(dst=dst)
