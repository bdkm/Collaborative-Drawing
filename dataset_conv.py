# Load a json file
# Preprocess in usual way
# Split canvas in arbitrary direction, d
# Reflect points to generate two records
# Save with class as d


import tensorflow as tf
import numpy as np
import ink_parser as parse
import random
import itertools
import math

"""
Loads quickdraw json file, parses contents and exports tfrecord holding
individual strokes
"""

def ink_array(xs, ys):
    a = np.array(xs)
    b = np.array(ys)
    c = np.zeros(len(xs))
    c[-1] = 1.0
    return np.column_stack((a,b,c))

def rotate(p, o, angle):
    qx = o[0] + math.cos(angle) * (p[0] - o[0]) - math.sin(angle) * (p[1] - o[1])
    qy = o[1] + math.sin(angle) * (p[0] - o[0]) + math.cos(angle) * (p[1] - o[1])
    return [qx, qy, p[2]]

def add_jitter(stroke, jitter_amount):
    if len(stroke) == 0:
        return stroke
    zer = np.zeros(stroke.shape)
    r = np.array([[random.uniform(-jitter_amount, jitter_amount) for _ in range(2)] for _ in range(stroke.shape[0])])
    stroke[:,0:2] += r
    return stroke

def write_ink_record(dest_name, number):
    filename = "%s.tfrecords" % (dest_name)
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(number):
        if i % 1000 == 0:
            print(i)

        class_index = 0
        size = random.randint(5, 25)

        # Line
        y = random.uniform(0.0, 1.0)
        l = random.uniform(0.1,1.0)
        x = random.uniform(0.0,1.0 - l)
        xs = [random.uniform(0.0,1.0) for _ in range(size)]
        xs_len = sum(xs)
        xs = [x / xs_len * l for x in xs]
        xs = list(itertools.accumulate(xs))
        ys = [y for _ in range(size)]

        inks = ink_array(xs,ys)

        if random.uniform(0.0, 1.0) > 0.5:
            # Corner
            class_index = 1

            corner_index = random.randint(2, size - 2)

            corner = inks[corner_index]
            inks_1 = inks[:corner_index]
            inks_2 = inks[corner_index:]

            inks_2 = [rotate(p, corner, math.pi / 2) for p in inks_2]
            inks = np.vstack((inks_1,inks_2))

        angle = random.uniform(0.0, 4.0) * math.pi / 2
        inks = np.array([rotate(p, [0.5,0.5], angle) for p in inks])

        #inks = add_jitter(inks, 0.005)

        inks = parse.normalize_and_compute_deltas(inks)

        features = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index])),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[size,3])),
                    'ink': tf.train.Feature(float_list=tf.train.FloatList(value=inks.flatten()))}

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())

    writer.close()

def main():
    write_ink_record('dataset/corners-train', 100000)
    write_ink_record('dataset/corners-eval', 10000)


if __name__ == '__main__':
    main()
