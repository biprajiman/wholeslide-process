# Copyright 2017 BICI2 Lab University of Florida
# Author: Manish Sapkota
# Some of the codes have been adopted from original Tensorflow examples
# ==============================================================================

"""Builds the UNET network."""
# pylint: disable=missing-docstring
# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import tensorflow as tf
import layers

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 572
NUM_CLASSES = 2  # foreground and background
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2000


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
DEFAULT_WEIGHT_DECAY = 0.0005

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))

def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def Unet(images):
    """Build the Unet model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        conv1_1 = layers.conv_relu(images, [3, 3, 3, 64], scope.name)
        _activation_summary(conv1_1)
    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
        conv1_2 = layers.conv_relu(conv1_1, [3, 3, 64, 64], scope.name)
        _activation_summary(conv1_2)
    # pool1
    pool1 = layers.max_pool(conv1_2, 'pool1', False)

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        conv2_1 = layers.conv_relu(pool1, [3, 3, 64, 128], scope.name)
        _activation_summary(conv2_1)
    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        conv2_2 = layers.conv_relu(conv2_1, [3, 3, 64, 128], scope.name)
        _activation_summary(conv2_2)
    # pool2
    pool2 = layers.max_pool(conv1_2, 'pool2', False)

    # conv3_1
    with tf.variable_scope('conv3_1') as scope:
        conv3_1 = layers.conv_relu(pool2, [3, 3, 128, 256], scope.name)
        _activation_summary(conv3_1)
    # conv3_2
    with tf.variable_scope('conv3_2') as scope:
        conv3_2 = layers.conv_relu(conv3_1, [3, 3, 256, 256], scope.name)
        _activation_summary(conv3_2)
    # pool3
    pool3 = layers.max_pool(conv3_2, 'pool3', False)

    # conv4_1
    with tf.variable_scope('conv4_1') as scope:
        conv4_1 = layers.conv_relu(pool3, [3, 3, 256, 512], scope.name)
        _activation_summary(conv4_1)
    # conv4_2
    with tf.variable_scope('conv4_2') as scope:
        conv4_2 = layers.conv_relu(conv4_1, [3, 3, 512, 512], scope.name)
        _activation_summary(conv4_2)
        conv4_2 = layers.dropout(conv4_2)
    # pool4
    pool4 = layers.max_pool(conv4_2, 'pool4', False)

    # conv5_1
    with tf.variable_scope('conv5_1') as scope:
        conv5_1 = layers.conv_relu(pool4, [3, 3, 512, 1024], scope.name)
        _activation_summary(conv5_1)
    # conv5_2
    with tf.variable_scope('conv5_2') as scope:
        conv5_2 = layers.conv_relu(conv5_1, [3, 3, 1024, 1024], scope.name)
        _activation_summary(conv5_2)
        conv5_2 = layers.dropout(conv5_2)

    # deconv1
    with tf.variabale_scope('deconv1') as scope:
        deconv1 = layers.deconv_relu(conv5_2, [2, 2, 1024, 512], scope.name)
        _activation_summary(deconv1)
    # crop concat
    with tf.variable_scope('crop_concat_1') as scope:
        crop_concat_1 = layers.crop_and_concat(conv4_2, deconv1) # doubles the feature map
    # conv6_1
    with tf.variable_scope('conv6_1') as scope:
        conv6_1 = layers.conv_relu(crop_concat_1, [3, 3, 1024, 512], scope.name)
        _activation_summary(conv6_1)
    # conv6_2
    with tf.variable_scope('conv6_2') as scope:
        conv6_2 = layers.conv_relu(conv6_1, [3, 3, 512, 512], scope.name)
        _activation_summary(conv6_2)

    # deconv2
    with tf.variabale_scope('deconv2') as scope:
        deconv2 = layers.deconv_relu(conv6_2, [2, 2, 512, 256], scope.name)
        _activation_summary(deconv1)
    # crop concat
    with tf.variable_scope('crop_concat_2') as scope:
        crop_concat_2 = layers.crop_and_concat(conv3_2, deconv2) # doubles the feature map
    # conv7_1
    with tf.variable_scope('conv7_1') as scope:
        conv7_1 = layers.conv_relu(crop_concat_2, [3, 3, 512, 256], scope.name)
        _activation_summary(conv7_1)
    # conv7_2
    with tf.variable_scope('conv7_2') as scope:
        conv7_2 = layers.conv_relu(conv7_1, [3, 3, 256, 256], scope.name)
        _activation_summary(conv7_2)

    # deconv3
    with tf.variabale_scope('deconv3') as scope:
        deconv3 = layers.deconv_relu(conv7_2, [2, 2, 256, 128], scope.name)
        _activation_summary(deconv3)
    # crop concat
    with tf.variable_scope('crop_concat_3') as scope:
        crop_concat_3 = layers.crop_and_concat(conv2_2, deconv3) # doubles the feature map
    # conv8_1
    with tf.variable_scope('conv8_1') as scope:
        conv8_1 = layers.conv_relu(crop_concat_3, [3, 3, 256, 128], scope.name)
        _activation_summary(conv8_1)
    # conv8_2
    with tf.variable_scope('conv8_2') as scope:
        conv8_2 = layers.conv_relu(conv8_1, [3, 3, 128, 128], scope.name)
        _activation_summary(conv8_2)

     # deconv4
    with tf.variabale_scope('deconv4') as scope:
        deconv4 = layers.deconv_relu(conv8_2, [2, 2, 128, 64], scope.name)
        _activation_summary(deconv4)
    # crop concat
    with tf.variable_scope('crop_concat_4') as scope:
        crop_concat_4 = layers.crop_and_concat(conv1_2, deconv4) # doubles the feature map
    # conv9_1
    with tf.variable_scope('conv9_1') as scope:
        conv9_1 = layers.conv_relu(crop_concat_4, [3, 3, 128, 64], scope.name)
        _activation_summary(conv9_1)
    # conv9_2
    with tf.variable_scope('conv9_2') as scope:
        conv9_2 = layers.conv_relu(conv9_1, [3, 3, 128, 64], scope.name)
        _activation_summary(conv9_2)

    # softmax_linear
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        softmax_linear = layers.conv_relu(conv9_2, [1, 1, 64, NUM_CLASSES], scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
