# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:27:18 2018

@author: Patrick
"""

import tensorflow as tf

learning_rate = 0.0001
epochs = 10
batch_size = 50

coef_num = 13
samples = 200

x = tf.placeholder(tf.float32, [None, coef_num, samples, 1])
y = tf.placeholder(tf.float32, [None, coef_num, samples, 1])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer

layer1 = create_new_conv_layer(x, 1, 32, [5, 5], [2, 2], name='layer1')