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


DEFAULT_WEIGHT_DECAY = 0.0
FLAGS = tf.app.flags.FLAGS

def variable_on_cpu(name, shape, initializer):

    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, w_decay=None):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(name,
                          shape,
                          tf.contrib.layers.xavier_initializer())

    if w_decay is None or w_decay <= 0.0:
        w_decay = DEFAULT_WEIGHT_DECAY

    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                   w_decay,
                                   name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

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

    kernel = variable_with_weight_decay('weights',
                                        shape=kernel_shape,
                                        w_decay=0.0)
    conv = tf.nn.conv2d(bottom,
                        kernel, [1, 1, 1, 1],
                        padding='SAME')
    biases = variable_on_cpu('biases',
                             [n_filters],
                             tf.constant_initializer(0.0))
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
    kernel = variable_with_weight_decay('weights',
                                        shape=kernel_shape,
                                        w_decay=0.0)

    print(scope_name+ ":")
    print(kernel.get_shape())
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
    biases = variable_on_cpu('biases',
                              [n_filters],
                              tf.constant_initializer(0.0))

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
    weights = variable_with_weight_decay('weights', shape=shape)
    biases = variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
    return tf.nn.relu(tf.matmul(bottom, weights) + biases, name=scope_name)

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

"""
Code source: https://github.com/shekkizh/FCN.tensorflow/blob/master/TensorflowUtils.py
"""
def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
