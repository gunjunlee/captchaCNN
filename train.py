import tensorflow as tf
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

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

with tf.device('/cpu:0'):

    x_cpu = tf.placeholder(tf.float32, [None, 784])
    x_img_cpu = tf.reshape(x_cpu, [-1, 28, 28, 1])
    y_cpu = tf.placeholder(tf.float32, [None, 10])

    W1_cpu = W1
    L1_conv_cpu = tf.nn.conv2d(x_img_cpu, W1_cpu, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu_cpu = tf.nn.relu(L1_conv_cpu)
    L1_cpu = tf.nn.max_pool(L1_relu_cpu, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

    W2_cpu = W2
    L2_conv_cpu = tf.nn.conv2d(L1_cpu, W2_cpu, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu_cpu = tf.nn.relu(L2_conv_cpu)
    L2_cpu = tf.nn.max_pool(L2_relu_cpu, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

    L2_reshape_cpu = tf.reshape(L2_cpu, [-1, 7*7*64])

    W3_cpu = W3
    b_cpu = b
    hyp_cpu = tf.matmul(L2_reshape_cpu, W3_cpu) + b_cpu

    cost_cpu = tf.nn.softmax_cross_entropy_with_logits(logits=hyp_cpu, labels=y_cpu)

    correct_cpu = tf.cast(tf.equal(tf.arg_max(hyp_cpu, 1), tf.arg_max(y_cpu, 1)), tf.float32)
    accuracy_cpu = tf.reduce_mean(correct_cpu)


train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()

epochs = 15
batch_size = 100
num_batch = int(mnist.train.num_examples / batch_size)

sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    for i in range(num_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x:batch_x, y:batch_y})
    a = sess.run(accuracy_cpu, feed_dict={x_cpu:mnist.test.images, y_cpu:mnist.test.labels})
    #tf.device('/cpu:0')
    #a = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print('accuracy: ', a)
    print(epoch)
    #tf.device('/gpu:0')

saver = tf.train.Saver()
saver.save(sess, 'model/train.ckpt')