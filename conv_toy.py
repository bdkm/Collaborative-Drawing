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
    points = points.reshape([int(points.shape[0] / 2),2])
    print(points)
    #points = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[1.0,0.5],[0.5,1.0],[0.5,0.5],[0.0,0.5]])
    x,y = np.array(points.T)
    deltas = compute_deltas(points)
    kernel = np.array([0.5,0,0.5,0])
    points = points.flatten()
    points_tensor = tf.constant(deltas,tf.float32)
    kernel_tensor = tf.constant(kernel,tf.float32)

    points_tensor = tf.reshape(points_tensor, [1,int(points.shape[0] / 2),2])
    kernel_tensor = tf.reshape(kernel_tensor, [2,2,1])
    print(points_tensor)
    print(kernel_tensor)
    """convolved = tf.nn.conv1d(
        points_tensor,
        kernel_tensor,
        stride=1,
        padding="SAME",
        name="conv1d_0")
    """
    convolved = tf.layers.conv1d(
        points_tensor,
        filters=1,
        kernel_size=2,
        kernel_initializer = tf.constant_initializer(kernel),
        activation=None,
        strides=1,
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="conv1d_1")
    #kernel_tensor = tf.get_variable('w', initializer=tf.to_float(kernel)),
    #mag = tf.abs(convolved)
    #sum = tf.reduce_sum(tf.transpose(tf.reshape(mag, [2,int(points.shape[0] / 2.0)])), axis=1)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        a = sess.run(points_tensor)
        b = sess.run(convolved)
        #c = sess.run(mag)
        #d = sess.run(sum)
        print(a)
        print(b)
        #print(c)
        #print(d)


    values = b.flatten().tolist()
    print(values)
    sizes = [abs(v) * 10 for v in values]

    plt.figure(figsize=(8,8))
    plt.plot(x,y,'k')
    for i in range(2, len(values)):
        if values[i] > 0:
            plt.scatter(x[i],y[i],s=sizes[i],facecolors='k', edgecolors='k')
        else:
            plt.scatter(x[i],y[i],s=sizes[i],facecolors='w', edgecolors='k')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
