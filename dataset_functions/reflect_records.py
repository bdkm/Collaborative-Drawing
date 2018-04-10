# Load a json file
# Preprocess in usual way
# Split canvas in arbitrary direction, d
# Reflect points to generate two records
# Save with class as d


import tensorflow as tf
import numpy as np
import ink_parser as parse
import random

"""
Loads quickdraw json file, parses contents and exports tfrecord holding
individual strokes
"""

def split_by_indicies(l,ss):
    ll = []
    start = 0
    for s in ss:
        ll.append(l[start:s + 1])
        start = s + 1
    return ll

def split_strokes(ink):
    # Separate strokes
    ss = ink[:,2]
    ss = np.nonzero(ss)[0]
    strokes = split_by_indicies(ink, ss)
    return strokes

def split_accross_center(stroke):
    is_left = (stroke < 0.5)[:,0]
    left = stroke[is_left]
    right = stroke[~is_left]
    if len(left) > 0:
        left[-1][2] = 1
    if len(right) > 0:
        right[-1][2] = 1
    return left, right

def reflect_accross_center(stroke):
    reflected = np.copy(stroke)
    reflected[:,0] = 1 - reflected[:,0]
    return reflected

def randomize_direction(stroke):
    if random.uniform(0.0, 1.0) < 0.5:
        stroke = stroke[::-1]
        if len(stroke) > 0:
            stroke[0][2] = 0.0
            stroke[-1][2] = 1
    return stroke

def connect(stroke1, stroke2):
    stroke2 = stroke2[::-1]
    if len(stroke1) > 0:
        stroke1[-1][2] = 0
    if len(stroke2) > 0:
        stroke2[0][2] = 0
        stroke2[-1][2] = 1
    return np.concatenate([stroke1, stroke2])

def add_jitter(stroke, jitter_amount):
    if len(stroke) == 0:
        return stroke
    zer = np.zeros(stroke.shape)
    r = np.array([[random.uniform(-jitter_amount, jitter_amount) for _ in range(2)] for _ in range(stroke.shape[0])])
    stroke[:,0:2] += r
    return stroke

def write_ink_record(source_names, dest_name, start, end):
    filename = "%s_reflected_h.tfrecords" % (dest_name)
    writer = tf.python_io.TFRecordWriter(filename)
    for source_name in source_names:
        print(source_name)
        with open(source_name) as f:
            lines = f.readlines()

            for i in range(start,end):
                if i % 1000 == 0:
                    print(i)
                l = lines[i]

                class_name, ink = parse.parse_json(l)
                ink = parse.reshape_ink(ink)
                ink = parse.normalise(ink)
                strokes = split_strokes(ink)

                for stroke in strokes:
                    left, right = split_accross_center(stroke)
                    left_reflect = reflect_accross_center(left)
                    right_reflect = reflect_accross_center(right)
                    #left = randomize_direction(left)
                    #right = randomize_direction(right)
                    #left_reflect = randomize_direction(left_reflect)
                    #right_reflect = randomize_direction(right_reflect)

                    lefts = connect(np.array(left), left_reflect)
                    rights = connect(np.array(right), right_reflect)

                    lefts = randomize_direction(lefts)
                    rights = randomize_direction(rights)

                    lefts = add_jitter(lefts, 0.02)
                    rights = add_jitter(rights, 0.02)

                    lefts = parse.normalize_and_compute_deltas(lefts)
                    rights = parse.normalize_and_compute_deltas(rights)
                    stroke = parse.normalize_and_compute_deltas(stroke)

                    left_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(lefts.shape))),
                                    'ink': tf.train.Feature(float_list=tf.train.FloatList(value=lefts.flatten()))}
                    right_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(rights.shape))),
                                    'ink': tf.train.Feature(float_list=tf.train.FloatList(value=rights.flatten()))}
                    stroke_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(stroke.shape))),
                                    'ink': tf.train.Feature(float_list=tf.train.FloatList(value=stroke.flatten()))}


                    left_example = tf.train.Example(features=tf.train.Features(feature=left_feature))
                    right_example = tf.train.Example(features=tf.train.Features(feature=right_feature))
                    stroke_example = tf.train.Example(features=tf.train.Features(feature=stroke_feature))
                    writer.write(left_example.SerializeToString())
                    writer.write(right_example.SerializeToString())
                    writer.write(stroke_example.SerializeToString())

        writer.close()

def main():
    write_ink_record(["dataset/json/full-simplified-camera.ndjson"], 'dataset/reflected', 10000, 100000)


if __name__ == '__main__':
    main()
