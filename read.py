import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import detection

with tf.device('/cpu:0'):

    x = tf.placeholder(tf.float32, [None, 784])
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    L1_conv = tf.nn.conv2d(x_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu = tf.nn.relu(L1_conv)
    L1 = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2_conv = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu = tf.nn.relu(L2_conv)
    L2 = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

    L2_reshape = tf.reshape(L2, [-1, 7*7*64])

    W3 = tf.get_variable("w3",  shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))
    hyp = tf.matmul(L2_reshape, W3) + b

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=hyp, labels=y)

    correct = tf.cast(tf.equal(tf.arg_max(hyp, 1), tf.arg_max(y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct)

sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, 'model/train.ckpt')

#print(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

for i in range(100):
    dir = 'samples/'+str(i)+'.png'
    n = detection.numberdetect(dir)
    x_data = np.multiply(n.firstnum_img(), 1/256)
    y_data = np.multiply(n.secondnum_img(), 1/256)
    if i == 58:
        plt.imshow(x_data)
        plt.show()
    x_data_1D = sess.run(tf.reshape(x_data, [-1, 784]))
    y_data_1D = sess.run(tf.reshape(y_data, [-1, 784]))
    print('i: ', i, end='')
    print(sess.run(tf.argmax(hyp, -1), feed_dict={x:x_data_1D}), end='')
    print(sess.run(tf.argmax(hyp, -1), feed_dict={x:y_data_1D}))
