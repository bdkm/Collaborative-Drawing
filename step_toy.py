import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compute_deltas(points):
    points[1:] = points[1:] - points[0:-1]
    return points

def plot_conv(points):
    points = np.array(points[0])
    print(points)
    num = int(points.shape[0] / 2)
    points = points.reshape([num,2])
    print(points)
    #points = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[1.0,0.5],[0.5,1.0],[0.5,0.5],[0.0,0.5]])
    x,y = np.array(points.T)
    deltas = compute_deltas(points)
    kernel = np.zeros(points.shape)
    kernel[0] = 1
    kernel[-1] = 1
    print(points.shape)
    print(kernel)
    points = points.flatten()
    points_tensor = tf.constant(deltas,tf.float32)
    #kernel_tensor = tf.constant(kernel,tf.float32)

    points_tensor = tf.reshape(points_tensor, [1,int(points.shape[0] / 2),2])
    #kernel_tensor = tf.reshape(kernel_tensor, [2,2,1])
    print(points_tensor)
    #print(kernel_tensor)

    repeat = tf.tile(points_tensor, tf.constant([1,2,1]))
    convolved = tf.layers.conv1d(
        repeat,
        filters=1,
        kernel_size=num,
        kernel_initializer = tf.constant_initializer(kernel),
        activation=None,
        strides=1,
        padding="valid",
        reuse=tf.AUTO_REUSE,
        name="conv1d_1")
    #kernel_tensor = tf.get_variable('w', initializer=tf.to_float(kernel)),
    #mag = tf.abs(convolved)
    #sum = tf.reduce_sum(tf.transpose(tf.reshape(mag, [2,int(points.shape[0] / 2.0)])), axis=1)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        s = points_tensor.shape
        print(s[1])
        #a = sess.run(points_tensor[:,::int(float(points_tensor.get_shape().as_list()[1]) / 2)])
        b = sess.run(repeat)
        c = sess.run(convolved)
        #d = sess.run(sum)
        #print(a)
        print(b)
        print(c)
        #print(d)
"""

    values = a.flatten().tolist()
    print(values)
    sizes = [abs(v) * 4 for v in values]

    plt.plot(x,y,'k')
    for i in range(2, len(values)):
        if values[i] > 0:
            plt.scatter(x[i],y[i],s=sizes[i],facecolors='k', edgecolors='k')
        else:
            plt.scatter(x[i],y[i],s=sizes[i],facecolors='w', edgecolors='k')
    plt.show()
"""
