<h1>何謂特徵？</h1>

前面我們介紹了 Filter 的操作，了解計算圖片的導數就會取得邊緣

在電腦視覺中我們也可以把這個過程稱為「抽取特徵」


所謂特徵，可以非常直覺的解釋為「圖片中最特別的地方」

可以是邊緣，輪廓，紋理等資訊

<h1>SIFT 特徵 - 概念</h1>

本章節要介紹的 SIFT 就是其中一種表徵 (appearance feature)

基於局部的外觀特徵，進一步考慮到圖片特徵的狀況

* 縮放不變性，旋轉不變性
* 光線與雜訊容忍度高


在 SIFT 演算法中可以了解到如何做關鍵點偵測，並如何抽出 SIFT 特徵來敘述關鍵點

**下圖：示意該圖片的 SIFT 特徵**

![SIFT 特徵](/repo/image/1596169294364.png)

<h1>SIFT 特徵 - 尺度不變性</h1>

![尺度不變性](/repo/image/1596169349105.png)

SIFT 主要考慮到的問題之一是尺度

以 corner detector (e.g. Harris) 為例，Filter 可以偵測到範圍以內的角落點，

但是同樣的 pattern 放大後以同樣的 Filter 去偵測就會失敗

<h1>SIFT 特徵 - 尺度空間極值偵測</h1>

SIFT 會基於邊緣檢測抽取特徵，但不是使用前面提過的 Sobel

概念上是 LoG 但是考慮到計算量 __使用 DoG 做邊緣檢測__


**Laplacian of Gaussian (LoG)**

* 先對圖片做 Gaussian Blur 再算二階導數取得邊緣

**Difference of Gaussian (DoG)**

* 圖片經過不同程度的縮放後計算出不同程度的 Gaussian Blur 最後合併得到一個 Gaussian Pyramid，其差值即為 DoG
* 結果可以視為 LoG 的約略值 (沒有做二階導數)

這邊討論的特徵主要是物體的邊緣

而二階導數是個適合的工具來找出邊緣，因此這邊才會以此討論 LoG 與 DoG

<h1>SIFT 特徵 - 尺度空間極值偵測 (DoG 尺度)</h1>

DoG 詳解 (如下圖所示)

* 圖片的一種 scale 稱為一個 octave
* 每種 scale 的圖片經過不同程度的 Gaussian Blur 在計算其差值
* 最後會得到下圖最後的 DoG (Gaussian Pyramid)

![DoG](/repo/image/1596169647324.png)

<h1>SIFT 特徵 - 尺度空間極值偵測 (極值偵測)</h1>

前面提到的 DoG 影像包含多種尺度，接著要針對每個 pixel 判斷是否為極值

判斷範圍 8+18 = 26

* 自己本身周遭的 8 個 pixel
* 同一個 scale 圖片但不同模糊尺度 相鄰位置共 9*2=18 個 pixel

假如該 pixel 為判斷範圍內的最大 / 最小值，則將其設為有興趣的關鍵點

![極值偵測](/repo/image/1596169975051.png)

<h1>SIFT 特徵 - 關鍵點定位</h1>

經過多尺度極值偵測之後，會得到許多候選的關鍵點，其中也包含許多噪音跟邊的關鍵點，需要更進一步根據周遭資訊來修正並過濾關鍵點


**鄰近資料差補**

* 主要根據相鄰資訊來修正極值的位置

**過濾不明顯關鍵點**

* 根據計算曲率來判斷是否為不明顯的關鍵點

**過濾邊緣關鍵點**

* 根據計算曲率來判斷是否為不明顯的關鍵點


Note: 這邊包含大量數學推導，不詳細解釋，如果有興趣可以參考最後的 reference

<h1>SIFT 特徵 - 方位定向</h1>

前面我們定義並過濾了許多關鍵點，但是關鍵點只有包含尺度跟位置

SIFT 還想要保有旋轉不變性，因此要給關鍵點定義一個方向

* 以每 10 度為單位計算周圍的梯度值
* 梯度值最大的方向當作是該關鍵點的主要方向

<h1>SIFT 特徵 - 關鍵點描述子</h1>

![關鍵點描述子](/repo/image/1596170310439.png)

賦與關鍵點位置，尺度，方向確保移動，縮放，旋轉的不變性

還需要額外建立描述子來確保不同光線跟視角也有不變性

* 描述子會正規化成 128 維的特徵向量
* 以關鍵點周圍 16*16 的區域共 4*4 的子區域，計算 8 個方向的直方圖，共 4*4*8 = 128 維的特徵向量

Note：每個關鍵點都會產生 128 維的特徵

而圖片會產生 N 個關鍵點，也就是會產生 (N, 128) 維度特徵

<h1>SIFT 實作</h1>

如果要透過 OpenCV 使用 SIFT 的話必須要額外安裝擴充的函式庫

為了避免版本問題，我們會指定安裝版本

pip install opencv-contrib-python==3.4.2.16

接著才有辦法使用 SIFT 來抽取特徵

```python
sift = cv2.xfeatures2d.SIFT_create() # 建立 SIFT 物件

keypoints = sift.detect(img_gray, None) # 抽取關鍵點

img_show = cv2.drawKeypoints(img_gray, keypoints, img)
```

<h1>知識點回顧</h1>

了解 SIFT 演算法的物理意義

* 從 LoG 變為 DoG 取得多尺度空間
* 透過極值找候選的關鍵點
* 透過主曲率過濾關鍵點
* 透過區域劃分決定方向並正規化成 128 維向量

---
---

<h1>特徵應用</h1>

當我們取得特徵之後，就代表我們已經有能力去識別關鍵點的特殊性

* 在這之後可以接到許多電腦視覺的任務
* 配對：判斷兩張圖片上相同物體的位置
* 辨識：判斷兩張圖片上是否有相同物體
* 全景圖：尋找兩張圖片的相同視角，再經過轉換合成全景圖
* ...

廣泛的說，SIFT 只是其中一種抽取特徵的方式，這邊會延續上一章節以 SIFT 為例介紹配對的應用。

<h1>特徵配對 Feature Matching - 任務目標</h1>

![特徵配對](/repo/image/1596179658728.png)

首先配對會有兩張圖片，其中一張是 query image，我們會各自在兩張上面抽取 SIFT 特徵。

<h1>SIFT 特徵 - 尺度不變性</h1>

配對會從兩張圖片中的關鍵點中，透過計算其特徵空間上的距離，

 若小於一個設定的閥值就視為是相同的特徵

在 SIFT 特徵的配對任務中，通常會使用 L2 norm 的方式計算

![L2 norm](/repo/image/1596179723141.png)

兩個 128 維向量根據上面公式計算可以得到一個距離

<h1>SIFT 特徵 - 尺度空間極值偵測</h1>

![尺度空間極值偵測](/repo/image/1596179810868.png)

簡單暴力的配對方法是逐一針對 query image 的關鍵點，對每個 train image 的關鍵點計算 L2 距離

* 取得距離最小的配對
* 取得 k 個最適合的配對


這邊為了確保配對的合適性，可以先在計算時取得 k 個配對，在根據距離去過濾不適合的配對

<h1>特徵配對 Feature Matching - ratio test</h1>

我們可以尋找 k=2 個最好的 match 方式，透過 ratio test 的方式來過濾一些不適當的配對，因為有時候 query 的關鍵點並不會出現在 train image

![尺度空間極值偵測](/repo/image/1596179835615.png)

根據補充資料的論文提到，建議比值設定在 0.7~0.8 比較好

<h1>特徵配對 Feature Matching - 實作</h1>

![實作1](/repo/image/1596179890760.png)

![實作2](/repo/image/1596180029208.jpg)

![實作3](/repo/image/1596179984230.png)

SIFT 在機器學習上的應用 (optional)

![機器學習上的應用1](/repo/image/1596180191354.png)

許多機器學習的任務一開始都要先經過抽取特徵的步驟，諸如 SIFT 等傳統電腦視覺的特徵只是其中一種方式，而近期非常熱門的深度學習則是另外一種抽取特徵的方式

後面再根據任務類型選擇要對特徵做甚麼處理，所以傳統特徵跟 model 抽的特徵，使用上是差不多的，e.g. 分類任務，我們就把抽完的特徵當作 input 輸入分類器。

![機器學習上的應用2](/repo/image/1596180223244.png)

SIFT 雖然可以做機器學習任務，但是實作上存在一些問題，因為演算法的關係，不保證所有圖片都會產生一樣維度的特徵

一般機器學習任務的 input 都要是同樣的維度，因此 SIFT 特徵必須做前處理

<h1>SIFT 在應用上的問題 (optional)</h1>

* SIFT 每一個特徵點的維度其實是一樣的，但每張圖片產生的特徵點個數不同，才會導致圖片的特徵維度不同
* 其中一種作法是做 Clustering，每一張圖片都取 n 個特徵點來固定圖片的特徵維度

缺點：

如果圖片太簡單導致部份圖片特徵太少就會失效，所以類似 MNIST 或是 CIFAR 等簡單的資料集就不太適合

<h1>知識點回顧</h1>

**了解特徵在電腦視覺的泛用性**

* 若有更適合的特徵可以替換 SIFT 直接接後面的流程

**了解特徵在配對任務中扮演的角色**

* 在多種電腦視覺任務中也很常出現透過距離來判斷兩個向量的相似度，這邊的物理意義就是判斷是否為相同特徵


