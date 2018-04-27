import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np


k1 = mpimg.imread('./Image_and_ImageData/key1.png')
k2 = mpimg.imread('./Image_and_ImageData/key2.png')
I = mpimg.imread('./Image_and_ImageData/I.png')
E = mpimg.imread('./Image_and_ImageData/E.png')
eprime = mpimg.imread('./Image_and_ImageData/Eprime.png')

plt.imshow(I, cmap="gray")
plt.show()


#print(k1.shape)  # (300, 400)
k1 = np.reshape(k1, [-1])  # (120000,)
k2 = np.reshape(k2, [-1])
I = np.reshape(I, [-1])
E = np.reshape(E, [-1])
eprime = np.reshape(eprime, [-1])

k1 = np.expand_dims(k1, axis=0)
k2 = np.expand_dims(k2, axis=0)
I = np.expand_dims(I, axis=0)
a = np.concatenate((k1, k2, I), axis=0)  # (3, 120000)

print(a.shape)
