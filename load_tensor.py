import tensorflow as tf
import numpy as np
import sys
import quickdraw_functions as qd
import ink_parser as ip
import estimator_shape as se
import input_functions as input
import estimator_conv as ce
"""
Loads contents of a quickdraw tensor into format for plotting
"""

"""Converts a list of deltas to coordinates"""
def roll_sum(l):
    for i in range(1, len(l)):
        l[i] = l[i - 1] + l[i]
    return l

"""Converts a list of coordinates to deltas"""
def unroll_sum(l):
    for i in range(1, len(l)):
        l[i] = l[i] - l[i - 1]
    return l

"""Splits a list into sublists determined by a list of indices"""
def split_by_indicies(l,ss):
    ll = []
    start = 0
    for s in ss:
        ll.append(l[start:s + 1])
        start = s + 1
    return ll

def something(sess, tensor, scale, center):
    f = sess.run(tensor)
    ink = f['ink']
    print("Class: %d" % f['class_index'])

    xs = ink[::3]
    ys = ink[1::3]
    ss = ink[2::3]

    ss = np.nonzero(ss)[0]

    xs = roll_sum(xs)
    ys = roll_sum(ys)

    xs = ((xs) * scale[0]) - (scale[0] / 2) + center[0]
    ys = ((ys) * scale[1]) - (scale[1] / 2) + center[1]

    xs = split_by_indicies(xs, ss)
    ys = split_by_indicies(ys, ss)

    strokes = []
    for x, y in zip (xs,ys):
        strokes.append([val for pair in zip(x, y) for val in pair])
    return strokes

"""Loads a tensor and formats it as [x1, y1, x2, y2, ...]"""
def load(filename, center, scale, index = (-1)):
    tensor = qd.get_quickdraw_tensor(filename)
    drawings = []
    try:
        with tf.Session() as sess:
            if index == -1:
                # Load all strokes in all tensors
                while True:
                    strokes = something(sess, tensor, scale, center)
                    drawings += strokes
            else:
                # Skip to index tensor and get all strokes from that tensor
                for _ in range(0,index):
                    f = sess.run(tensor)
                strokes = something(sess, tensor, scale, center)
                drawings += strokes
    except tf.errors.OutOfRangeError:
        print("Finished")
    return drawings

def class_to_color(c):
    mapping = {
        0: (0,0,1), # squiggle
        1: (0,1,0), # circle
        2: (0,1,0), # octagon
        3: (0,1,0), # hexagon
        4: (1,0,0), # square
        5: (1,1,0), # triangle
        6: (1,0,1)  # line
    }
    return mapping[c]

def classify(ink):
    batch_size = len(ink)
    classifier = ce.get_classifier(batch_size)
    predictions = classifier.predict(input_fn=lambda:input.ink_dataset(ink, batch_size))
    print(predictions)
    colors = []
    for p in predictions:
        print(p['classes'])
        print(p['probabilities'])
        #colors.append(class_to_color(p))
    return colors
