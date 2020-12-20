import numpy as np
from matplotlib import pyplot as plt

""" Working directory: CupoyLearning
損失函數是描述模型預測出來的結果和實際的差異的依據
YOLO 損失函數的設計包含物件位置的定位與物件類別辨識
YOLO損失函數透過超參數設定模型有不同的辨識能力

作業

仔細觀察，bbox 寬高計算損失方式和 bbox 中心計算損失方式有哪邊不一樣嗎? 為什麼要有不同的設計?

Ans: 
寬高損失計算時，先開了根號才相減。比起較小的物體，較大的物體在估計寬高時，與實際寬高的誤差即使一樣大，計算而出的損失也比較小。
而中心損失計算的方式，若預測值與實際值的誤差一樣大，計算而出的損失也會一樣。
"""


def drawLine(x, line="b-"):
    y = np.power(x, 0.5)
    plt.plot([x, x], [0, y], line)


# 寬高損失計算
loss = 2
small_x = 1
small_x_hat = small_x + loss
drawLine(small_x, "r-")
drawLine(small_x_hat, "r-")

big_x = 7
big_x_hat = big_x + loss
drawLine(big_x, "r-")
drawLine(big_x_hat, "r-")

small_y = np.power(small_x, 0.5)
plt.plot([small_x, small_x_hat], [small_y, small_y], "g-")
big_y = np.power(big_x, 0.5)
plt.plot([big_x, big_x_hat], [big_y, big_y], "g-")

x = np.linspace(0, 10)
y = np.power(x, 0.5)
plt.plot(x, y, "b-")
plt.show()
