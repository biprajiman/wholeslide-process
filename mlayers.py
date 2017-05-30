'''
Author: Manish Sapkota
Created: 05-12-2017
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=invalid-name


FLAGS = tf.app.flags.FLAGS

def weight_variable(name, shape):
    var = tf.get_variable(name,
                          shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return var

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv_relu(bottom, kernel_shape, scope_name):
    """ Helper to create convolution with relu non linearity
    Args:
        bottom: input feature map or the original image
        kernel_shape: shape of the convolution kernel to use
        scope_name: name of the variable
    Returns:
        Variable Tensor
    """
    n_filters = kernel_shape[-1]

    kernel = weight_variable('weights',
                                        shape=kernel_shape)
    conv = tf.nn.conv2d(bottom,
                        kernel, [1, 1, 1, 1],
                        padding='SAME')
    biases = bias_variable('biases',
                             [n_filters])

    pre_activation = tf.nn.bias_add(conv, biases)

    return tf.nn.relu(pre_activation, name=scope_name)

def deconv_relu(bottom, kernel_shape, scope_name, padding='VALID', output_shape=None):
    """ Helper to create deconvolution (transpose convolution) with relu non linearity
    Args:
        imput_images: input feature map or the original image
        kernel_shape: shape of the deconvolution kernel to use (h, w, out_channels, in_channels)
        output_shape: shape of the desired output (batch, h, w, out_channels)
        scope_name: name of the variable
    Returns:
        Variable Tensor
    """
    kernel = weight_variable('weights',
                              shape=kernel_shape)

    dyn_input_shape = tf.shape(bottom)
    n_filters = kernel_shape[2]

    # extract batch-size like as a symbolic tensor to allow variable size
    batch_size = dyn_input_shape[0]
    stride_h = 2
    stride_w = 2
    assert padding in {'SAME', 'VALID'}
    if padding is 'SAME':
        out_h = dyn_input_shape[1] * stride_h # stride width and height used 2 [1, 2, 2, 1]
        out_w = dyn_input_shape[2] * stride_w # stride width and height used 2 [1, 2, 2, 1]
    elif padding is 'VALID':
        out_h = (dyn_input_shape[1] - 1) * stride_h + kernel_shape[0]
        out_w = (dyn_input_shape[2] - 1) * stride_w + kernel_shape[1]

    output_shape = tf.stack([batch_size, out_h, out_w, n_filters])

    # print output_shape
    conv = tf.nn.conv2d_transpose(bottom, kernel,
                                  output_shape,
                                  strides=[1, stride_h, stride_w, 1],
                                  padding='SAME')
    biases = bias_variable('biases',
                           [n_filters])

    pre_activation = tf.nn.bias_add(conv, biases)
    return tf.nn.relu(pre_activation, name=scope_name)

def max_pool(bottom, name):
    """ Helper to create max pooling layer of default size and stride
    Args:
        bottom: bottom input tensor
        name: name for the layer
        debug: not sure copied code from somewhere. Yet to find out how it works
    Returns:
        Variable Tensor
    """
    pool = tf.nn.max_pool(bottom,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID',
                          name=name)
    return pool

def dropout(bottom, keep_prob=0.5):
    """ Helper to add the dropout to the network """
    return tf.nn.dropout(bottom, keep_prob)

def crop(bottom1, bottom2):
    """ Function to crop and concat the bottom1 features to match bottom2 similar to crop concat of unet"""
    b1_shape = tf.shape(bottom1)
    b2_shape = tf.shape(bottom2)

    # offsets for the top left corner of the crop
    offsets = [0, (b1_shape[1] - b2_shape[1]) // 2,
               (b1_shape[2] - b2_shape[2]) // 2, 0]
    size = [-1, b2_shape[1], b2_shape[2], -1]
    return tf.slice(bottom1, offsets, size)

def crop_and_concat(bottom1, bottom2):
    """ Function to crop and concat the bottom1 features to match bottom2 similar to crop concat of unet"""
    b1_shape = tf.shape(bottom1)
    b2_shape = tf.shape(bottom2)

    # offsets for the top left corner of the crop
    offsets = [0, (b1_shape[1] - b2_shape[1]) // 2,
               (b1_shape[2] - b2_shape[2]) // 2, 0]
    size = [-1, b2_shape[1], b2_shape[2], -1]
    b1_crop = tf.slice(bottom1, offsets, size)
    return tf.concat([b1_crop, bottom2], 3)

def full_connection(bottom, shape, scope_name):
    """Helper to create a fully connected layer"""
    weights = weight_variable('weights', shape=shape)
    biases = bias_variable('biases', [shape[-1]])
    return tf.nn.relu(tf.matmul(bottom, weights) + biases, name=scope_name)

#https://github.com/jakeret/tf_unet/blob/master/tf_unet/layers.py
def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
    return tf.div(exponential_map, evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map, tensor_sum_exp)

def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")
    # return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
