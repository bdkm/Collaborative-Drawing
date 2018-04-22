import tensorflow as tf
import numpy as np
import ink_parser as ip
import math

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

def rotate_tensor(xs, ys, angle):
    xo = xs - 0.5
    yo = ys - 0.5

    qx = 0.5 + math.cos(angle) * xo - math.sin(angle) * yo
    qy = 0.5 + math.sin(angle) * xo + math.cos(angle) * yo
    return qx, qy

def plot_activations(inks):
    tf.reset_default_graph()
    # Initialise data
    inks = array_rep_to_ink_rep(inks)
    lengths = [len(i) for i in inks]
    inks = [ip.normalize_and_compute_deltas(ink) for ink in inks]
    inks = [ink.tolist() for ink in inks]
    max_len = len(max(inks,key=len))
    inks = [pad(ink, max_len) for ink in inks]

    lengths = tf.constant(lengths, tf.int32)
    inks = tf.constant(inks, tf.float32)

    a = reflectional_symmetry_recogniser(inks, math.pi/2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(a)
        i = sess.run(inks)
        print(a)
        print(i)

def reflectional_symmetry_recogniser(inks, angle):
    sliced = tf.slice(inks, tf.constant([0,1,0]), tf.constant([-1,-1,2]))
    xs = tf.slice(sliced, tf.constant([0,0,0]), tf.constant([-1,-1,1]))
    ys = tf.slice(sliced, tf.constant([0,0,1]), tf.constant([-1,-1,1]))
    xs, ys = rotate_tensor(xs, ys, angle)
    xr = tf.reverse(xs, [1])
    yr = tf.reverse(ys, [1])
    xn = xr * (-1)

    sx = tf.concat([xs,xn], axis=2)
    sy = tf.concat([ys,yr], axis=2)
    sx = tf.reduce_sum(sx, [2])
    sy = tf.reduce_sum(sy, [2])
    sx = abs(sx)
    sy = abs(sy)

    sum = tf.reduce_mean(tf.concat([sx,sy], axis=1), [0,1])
    return sum
