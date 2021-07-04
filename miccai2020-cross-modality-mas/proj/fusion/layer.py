'''
covlution layer，pool layer，initialization。。。。
'''
import tensorflow as tf
import numpy as np


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        initial = tf.random_uniform(shape, -init_range, init_range)
        return tf.Variable(initial)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)


# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 3D convolution
def conv3d(x, W):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
    return conv_3d


# 3D deconvolution
def deconv3d(x, W):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 2, 2, 1, 1], padding='SAME')


# Max Pooling
def max_pool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 1, 1], strides=[1, 2, 2, 1, 1], padding='SAME')


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)
