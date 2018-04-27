ML2018_410521233
===
---
## Image Decryption by a Single-Layer Neural Network
#### A. 準備訓練樣本
* 讀取圖片
  matplotlib.image.imread(image path)
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

  image = numpy.reshape(image, [-1])
* 將 k1, k2, I 結合成 shape=[3, 120000] 的矩陣 a

  a = numpy.contcatenate((k1, k2, I), axis=0)

#### B. 使用的參數
* 梯度下降

  learning rate = 0.01

  tf.train.GradientDescentOptimizer(0.01)
* loss 函數

  loss = mean((e-E)^2^) 
  
  loss = tf.reduce_mean(tf.square(tf.subtract(e, E)))


#### C. 求得的向量 W
W = [w1, w2, w3]

W = [0.24547063, 0.65446323, 0.10165172]


#### D. 解碼後的 𝐸'
![](https://raw.githubusercontent.com/free00000000000/ML2018_410521233/master/result.png)

#### E. 遇到的問題

#### F. 學到了什麼
