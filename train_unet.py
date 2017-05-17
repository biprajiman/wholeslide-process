"""
Copyright 2017 BICI2 Lab University of Florida
Author: Pingjun 
Modified: Manish Sapkota
Some of the codes have been adopted from original Tensorflow examples
"""
# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylin: disable=print
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from termcolor import colored
import tensorflow as tf

# Adding local Keras
HOME_DIR = os.path.expanduser('~')
keras_version = 'keras_pingpong'
KERAS_PATH = os.path.join(HOME_DIR, 'Github', keras_version)
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))

import keras
import shutil
import time
import unet as u
from data_gen import data_weighted_loader

FLAGS = tf.app.flags.FLAGS


def set_tf_flags():
    # image parameters
    tf.app.flags.DEFINE_integer('input_rows', 512, """Input image height""")
    tf.app.flags.DEFINE_integer('input_cols', 512, """Input image width""")
    tf.app.flags.DEFINE_integer('input_channel', 3, """Input image channels""")
    tf.app.flags.DEFINE_integer('num_classes', 2, """Output classes""")

    # training parameters
    tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
    tf.app.flags.DEFINE_integer('num_epoch', 10)
    tf.app.flags.DEFINE_integer('num_examples_per_epoch', 20000)
    tf.app.flags.DEFINE_integer('batch_size', 5)
    tf.app.flags.DEFINE_float('initial_learning_rate', 0.1)
    tf.app.flags.DEFINE_float('lr_decay', 0.1)
    tf.app.flags.DEFINE_float('moving_average_decay', 0.9999)
    tf.app.flags.DEFINE_float('weight_decay', 0.0005)
    tf.app.flags.DEFINE_integer('decay_num_epoch', 1)
    tf.app.flags.DEFINE_string('training_dir', '../BladderData/Segmentation/')
    tf.app.flags.DEFINE_string('checkpoint_dir', './Checkpoints')
    tf.app.flags.DEFINE_string('log_dir', './Logs')
    tf.app.flags.DEFINE_string('model_name', 'Unet')
    tf.app.flags.DEFINE_integer('seed', 1234)

def train_unet():
    # Config gpu for session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.set_session(sess)

    # Setting image generator
    train_generator, _ = data_weighted_loader(
        FLAGS.training_dir, FLAGS.batch_size)
    img = tf.placeholder(tf.float32, shape=(
        None, FLAGS.input_rows, FLAGS.input_cols, FLAGS.input_channel))
    label = tf.placeholder(tf.float32, shape=(
        None, FLAGS.input_rows, FLAGS.input_cols, 1))
    weights = tf.placeholder(tf.float32, shape=(
        None, FLAGS.input_rows, FLAGS.input_cols, 1))

    # Model
    unet_pred = u.Unet(img)

    # define optimzer
    global_step = tf.Variable(0, name='global_step', trainable=False)

    loss_ori = u.loss(logits=unet_pred, labels=label)
    loss_mask = tf.multiply(loss_ori, weights)
    unet_loss = tf.reduce_mean(loss_mask) * 1.0 / tf.reduce_mean(weights)
    train_op = u.train(total_loss=unet_loss, global_step=global_step)

    unet_pred = tf.nn.softmax(unet_pred)
    unet_acc = tf.reduce_mean(u.dice_coef(label, unet_pred))

    # define summary for tensorboard
    summary_merged = tf.summary.merge_all()

    # define saver
    if os.path.exists(FLAGS.log_dir):
        shutil.rmtree(FLAGS.log_dir)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    saver = tf.train.Saver()

    # Training Begins
    tot_iter = FLAGS.num_minibatch * FLAGS.num_epoch
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        # restore from a checkpoint if exists
        if u.load_model(sess, saver, FLAGS.checkpoint_dir):
            print(" [*] Load previous model success.")
        else:
            print(" [*] No previous model found.")

        print("Training Begin:")

        start_step = global_step.eval()
        start_t = time.time()
        for ibatch in range(start_step + 1, start_step + tot_iter + 1):
            x_batch, y_batch, weight, _ = train_generator.next()

            feed_dict = {img: x_batch, label: y_batch, weights: weight}
            _, loss, summary, dice_score = sess.run(
                [train_op, unet_loss, summary_merged, unet_acc], feed_dict=feed_dict)

            global_step.assign(ibatch).eval()
            train_writer.add_summary(summary, ibatch)

            if ibatch % 10 == 0:
                time_elapsed = time.time() - start_t
                print ('epoch/batch:{:3d}/{:5d}, loss={:0.2f}, dice_score={:0.2f}, takes {:0.2f}s for {:3d} images'.format(
                    int(ibatch / FLAGS.num_minibatch), int(ibatch % FLAGS.num_minibatch), loss, dice_score, time_elapsed, FLAGS.batch_size * 10))
                start_t = time.time()
            if ibatch % FLAGS.num_minibatch == 0:
                # saving checkpoint every epoch
                u.save_model(sess, saver, FLAGS.checkpoint_dir,
                             FLAGS.model_name, global_step)
                print ('save a checkpoint at ' +
                       FLAGS.checkpoint_dir + '-' + str(ibatch))
    print("Training Finish.")

if __name__ == "__main__":
    print(colored("Tensorflow version: {}".format(tf.__version__), 'red'))
    print(colored("Keras version: {}".format(keras.__version__), 'red'))
    set_tf_flags()
    train_unet()
