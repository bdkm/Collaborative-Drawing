import numpy as np
import tensorflow as tf

def compute_deltas(points):
    points[1:] = points[1:] - points[0:-1]
    return points

def plot_conv(points):
    points = np.array(points[0])
    print(points)
    points = points.reshape([1,int(points.shape[0] / 2),2])
    print(points)
    #points = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[1.0,0.5],[0.5,1.0],[0.5,0.5],[0.0,0.5]])
    #x,y = np.array(points.T)
    deltas = compute_deltas(points)
    points = points.flatten()
    points_tensor = tf.constant(deltas,tf.float32)
    length_tensor = tf.constant(deltas.shape)

    print(deltas)
    print(deltas.shape[1])


    cell = tf.nn.rnn_cell.BasicLSTMCell
    # Initialise cell for each layer with num_nodes number of units
    cells_fw = [cell(2) for _ in range(2)]
    # Also initialise for backpropogation
    cells_bw = [cell(2) for _ in range(2)]
    # Stack LSTM layers for forward and backpropogation
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=points_tensor,
        dtype=tf.float32,
        scope="rnn_classification")

    # Zero out regions where sequences have no data (and so LSTM is just thinking)
    mask = tf.tile(tf.expand_dims(tf.sequence_mask([deltas.shape[1]], tf.shape(outputs)[1]), 2), [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
    # Produce fixed length embedding by summing outputs of LSTM
    outputs = tf.reduce_sum(zero_outside, axis=1)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        a = sess.run(points_tensor)
        b = sess.run(outputs)
        #c = sess.run(mag)
        #d = sess.run(sum)
        print(a)
        print(b)
        #print(c)
        #print(d)

"""
    values = b.flatten().tolist()
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
