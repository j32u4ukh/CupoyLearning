import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

"""Working directory: CupoyLearning"""

path = "image/Dog.JPG"

# 讀入照片
image = cv2.imread(path)

# 因為CV2會將照片讀成BGR，要轉回來
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
先設 BBOX 格式為 [X, Y, W, H]

也就是左上角那一點的座標以及 BBOX 的寬和高
"""

# X, Y, W, H
g_x, g_y, g_w, g_h = 1900, 700, 1800, 1800
p_x, p_y, p_w, p_h = 1800, 800, 1500, 1500

plt.rcParams['figure.figsize'] = (20, 10)
fig, ax = plt.subplots(1)

# 畫出圖片
ax.imshow(image)

# 畫 BBOX-Ground_Truth
rect_ground_truth = patches.Rectangle((g_x, g_y), g_w, g_h, linewidth=3, edgecolor='b', facecolor='none')
ax.text(1900, 700, 'Ground Truth', withdash=True, size=20)

# 畫 BBOX-Prediction
rect_region_proposal = patches.Rectangle((p_x, p_y), p_w, p_h, linewidth=3, edgecolor='r', facecolor='none', )
ax.text(1800, 800, 'Region_Proposal', withdash=True, size=20)

# Add the patch to the Axes
ax.add_patch(rect_ground_truth)
ax.add_patch(rect_region_proposal)

plt.show()

t_x = (g_x - p_x) / p_w
t_y = (g_y - p_y) / p_h
t_w = np.log10(g_w / p_w)
t_h = np.log10(g_h / p_h)

print('x 偏移量：', t_x)
# x 偏移量： 0.06666666666666667

print('y 偏移量：', t_y)
# y 偏移量： -0.06666666666666667

print('w 縮放量：', t_w)
# w 縮放量： 0.07918124604762482

print('h 縮放量： ', t_h)
# h 縮放量： 0.07918124604762482

"""假設 Predict 值 dx, dy, dw, dh"""

dx, dy, dw, dh = [0.05, -0.05, 0.12, 0.17]

loss = np.sum(np.square(np.array([dx, dy, dw, dh]) -
                        np.array([t_x, t_y, t_w, t_h])))
print("Loss值：", loss)
# Loss值： 0.010469772299242164
