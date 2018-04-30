import numpy as np
import tensorflow as tf
import math


def rotate(ink, angle):
    def cum_sub(l):
        offset = l[:-1]
        offset = tf.concat([[[0,0,0]],offset], 0)
        offset = tf.negative(offset)
        return tf.add(offset,l)
    sin = math.sin(angle)
    cos = math.cos(angle)

    ink = tf.cumsum(ink)
    xs = ink[:,0]
    ys = ink[:,1]
    xo = (xs - 0.5)
    yo = (ys - 0.5)
    xs = 0.5 + cos * xo- sin * yo
    ys = 0.5 + sin * xo + cos * yo
    result = tf.transpose(tf.concat([[xs],[ys],[ink[:,2]]], 0))
    return cum_sub(result)

def rotate_batch(batch, angle):
    return tf.map_fn(lambda b: rotate(b, angle), batch)
"""
ink = tf.constant([[[0.0,0.0,0],[0.5,0.0,0],[0.5,0.0,0],[0.5,0.5,1]],[[0.0,0.5,0],[0.5,0.0,0],[0.5,0.0,0],[0.5,0.5,1]]], tf.float32)
angle = math.pi / 2

with tf.Session() as sess:
    a = sess.run(rotate_batch(ink, angle))
    print(a)
"""
