import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np


k1 = mpimg.imread('./Image_and_ImageData/key1.png')
k2 = mpimg.imread('./Image_and_ImageData/key2.png')
I = mpimg.imread('./Image_and_ImageData/I.png')
E = mpimg.imread('./Image_and_ImageData/E.png')
eprime = mpimg.imread('./Image_and_ImageData/Eprime.png')

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

# w = [w1, w2, w3]
w = tf.Variable(tf.ones(shape=[1, 3]))

e = tf.matmul(w, a)

loss = tf.reduce_mean(tf.square(tf.subtract(e, E)))

optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

step = 1
while sess.run(loss) > 1e-5:
    sess.run(train)
    if step % 100 == 0:
        print('step', step)
        print('loss', sess.run(loss))
    step += 1

W = sess.run(w)

print(W)

'''
img = 

img = np.reshape(img, [300, 400])
plt.imshow(img)
plt.show()
'''
