import tensorflow as tf
import numpy as np
import ink_parser as ip

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""Ink shaping"""
def pad(list,length):
    list += [[0,0,0] for _ in range(length - len(list))]
    return list

def array_rep_to_ink_rep(ink):
    def reshape_2d(list):
        list = np.array(list)
        return list.reshape(-1,2)

    ink = [reshape_2d(stroke) for stroke in ink]

    def stroke_delimiter(height):
        zeros = np.zeros((height,1))
        zeros[-1] = 1
        return zeros

    ink = [np.hstack((stroke,stroke_delimiter(stroke.shape[0]))) for stroke in ink]
    return ink

def plot_activations(inks):
    tf.reset_default_graph()
    # Initialise data
    height = 5
    width = len(inks)
    inks = array_rep_to_ink_rep(inks)
    lengths = [len(i) for i in inks]
    inks = [ip.normalize_and_compute_deltas(ink) for ink in inks]
    inks = [ink.tolist() for ink in inks]
    max_len = len(max(inks,key=len))
    inks = [pad(ink, max_len) for ink in inks]

    lengths = tf.constant(lengths, tf.int32)
    inks = tf.constant(inks, tf.float32)

    # Create some variables.
    convolved1 = tf.layers.conv1d(
        inks,
        filters=48,
        kernel_size=5,
        activation=None,
        strides=1,
        padding="same",
        name="conv1d_0")

    convolved2 = tf.layers.conv1d(
        convolved1,
        filters=64,
        kernel_size=5,
        activation=None,
        strides=1,
        padding="same",
        name="conv1d_1")

    convolved3 = tf.layers.conv1d(
        convolved2,
        filters=96,
        kernel_size=3,
        activation=None,
        strides=1,
        padding="same",
        name="conv1d_2")

    mask = tf.sequence_mask(lengths, tf.shape(convolved3)[1])
    mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, tf.shape(convolved3)[2]])
    zero_outside = tf.where(mask, convolved3, tf.zeros_like(convolved3))
    final_state = tf.reduce_sum(zero_outside, axis=1)
    logits = tf.layers.dense(final_state, 2)
    predictions = tf.argmax(logits, axis=1)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "models/reflected_model_conv/model.ckpt-500000")
        print("Model restored.")
        # Check the values of the variables
        i = 0
        x = sess.run(inks)
        a = sess.run(convolved1)
        aa = sess.run(convolved2)
        aaa = sess.run(convolved2)
        b = sess.run(zero_outside)
        c = sess.run(final_state)
        d = sess.run(logits)
        e = sess.run(predictions)
        print(d)
        for j in range(len(x)):
          plt.subplot(height, width, (i * width) + j + 1)
          plt.imshow(x[j])
        i = i + 1
        for j in range(len(a)):
          plt.subplot(height, width, (i * width) + j + 1)
          plt.imshow(a[j])
        i = i + 1
        for j in range(len(b)):
          plt.subplot(height, width, (i * width) + j + 1)
          plt.imshow(b[j])
        plt.subplot(height, 1, 4)
        plt.imshow(c)
        plt.subplot(height, 1, 5)
        plt.imshow(d)

        print(a)
        print(b)
        print(c)
        print(d)
        plt.show()
        return e
