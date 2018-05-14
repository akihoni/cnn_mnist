# coding:utf-8

import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
CONV1_SIZE = 5
CONV1_NUMBER = 32
CONV2_SIZE = 5
CONV2_NUMBER = 64
FC_LAYER1 = 512
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, trainable, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_NUMBER], regularizer)
    conv1_b = get_bias([CONV1_NUMBER])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_NUMBER, CONV2_NUMBER], regularizer)
    conv2_b = get_bias([CONV2_NUMBER])
    conv2 = conv2d(conv1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 对提取出的特征进行整形 之后再喂入fc网络
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_LAYER1], regularizer)
    fc1_b = get_bias([FC_LAYER1])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if trainable:
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_LAYER1, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b

    return y
