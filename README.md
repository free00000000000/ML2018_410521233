ML2018_410521233
===
---
## Image Decryption by a Single-Layer Neural Network
#### A. 準備訓練樣本
* 讀取圖片

  `matplotlib.image.imread(image path)`
  * k1
  
    ![](https://github.com/free00000000000/ML2018_410521233/blob/master/Image_and_ImageData/key1.png?raw=true)
  * k2
  
    ![](https://github.com/free00000000000/ML2018_410521233/blob/master/Image_and_ImageData/key2.png?raw=true)
  * I
  
    ![](https://github.com/free00000000000/ML2018_410521233/blob/master/Image_and_ImageData/I.png?raw=true)
  * E
  
    ![](https://github.com/free00000000000/ML2018_410521233/blob/master/Image_and_ImageData/E.png?raw=true)
  * eprime
  
    ![](https://github.com/free00000000000/ML2018_410521233/blob/master/Image_and_ImageData/Eprime.png?raw=true)
* 將圖片由 shape=[300, 400] 的矩陣轉成 shape=[120000] 的矩陣

  `image = numpy.reshape(image, [-1])`
* 將 k1, k2, I 結合成 shape=[3, 120000] 的矩陣 a

  `a = numpy.contcatenate((k1, k2, I), axis=0)`

#### B. 使用的參數
* MaxIterLimit: 3556
* 𝜖: 1e-5
* 𝛼: 0.01


#### C. 求得的向量 W
W = [w1, w2, w3]

W = [0.24547063, 0.65446323, 0.10165172]


#### D. 解碼後的 𝐸'
![](https://raw.githubusercontent.com/free00000000000/ML2018_410521233/master/result.png)

#### E. 遇到的問題
* 印出來的圖片不是黑白的，最後在 imshow 中加上參數 cmap='gray' 就解決了。
* 不熟悉 tensorflow，參考了許多文件。

#### F. 學到了什麼
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先學到了 Gradient descent 的概念，並運用這個概念求得 w1, w2, w3。
  Gradient descent 利用了微分的方法不斷更新 w 去最小化 loss 函數的值，最後能得到一個誤差極小的 w。
  在推導的過程中，重新練習了荒廢許久的微積分，以及線性代數。以下是大略推導過程:<br>
  `h(k1, k2, I) = w1*k1 + w2*k2 + w3*I`&nbsp;&nbsp;&nbsp;&nbsp;(hypothesis)<br>
  `L(w) = 1/2m*Σ(h(k1, k2, I) - E)^2`&nbsp;&nbsp;&nbsp;&nbsp;(L: loss 函數;&nbsp;&nbsp; m: 總數)<br>
  `w' = w - 𝛼*(d/dw)*L(w)`&nbsp;&nbsp;&nbsp;&nbsp;(w': 更新後的 w;&nbsp;&nbsp; 𝛼: 學習率;&nbsp;&nbsp; L: loss 函數)<br>
  `w1' = w1 - a(1/m*Σ(h(k1, k2, I) - E)*k1)`<br>
  `w2' = w2 - a(1/m*Σ(h(k1, k2, I) - E)*k2)`<br>
  `w3' = w3 - a(1/m*Σ(h(k1, k2, I) - E)*I)`<br><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最後用 tensorflow 實作 linear regression 實踐了這個作業，
  過程中學到許多 tensorflow 的基本用法，發現到若是能妥善運用，tensorflow 會是個十分好用的工具。
