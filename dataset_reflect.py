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

def resample(stroke, num):
    sample = random.sample(range(len(stroke)), num)
    stroke = np.array([stroke[i] for i in sample])
    return stroke

def write_ink_record(source_names, dest_name, start, end):
    filename = "%s.tfrecords" % (dest_name)
    writer = tf.python_io.TFRecordWriter(filename)
    num_sym = 0
    syms = 0
    num_not = 0
    nots = 0
    for source_name in source_names:
        print(source_name)
        with open(source_name) as f:
            lines = f.readlines()

            for i in range(start,end):
                if i % 1000 == 0:
                    print(i)
                l = lines[i]

                class_name, ink, recognised = parse.parse_json(l)
                ink = parse.reshape_ink(ink)
                ink = parse.normalise(ink)
                strokes = split_strokes(ink)

                for stroke in strokes:
                    if stroke.shape[0] == 0:
                        continue
                    left, right = split_accross_center(stroke)

                    #lefts = add_jitter(lefts, 0.02)
                    #rights = add_jitter(rights, 0.02)

                    added = 0
                    if (random.uniform(0.0, 1.0) > 0.5):
                        if (left.shape[0] > 0):
                            if (left.shape[0] > (stroke.shape[0] / 2)):
                                left = resample(left, int(stroke.shape[0] / 2))
                            else:
                                stroke = resample(stroke, (left.shape[0] * 2))

                            left_reflect = reflect_accross_center(left)
                            lefts = connect(np.array(left), left_reflect)
                            lefts = randomize_direction(lefts)
                            added = 1

                            num_sym += lefts.shape[0]
                            syms += 1
                            lefts = parse.normalize_and_compute_deltas(lefts)
                            left_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                                            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(lefts.shape))),
                                            'ink': tf.train.Feature(float_list=tf.train.FloatList(value=lefts.flatten()))}
                            left_example = tf.train.Example(features=tf.train.Features(feature=left_feature))
                            writer.write(left_example.SerializeToString())
                    else:
                        if right.shape[0] > 0:
                            if (right.shape[0] > (stroke.shape[0] / 2)):
                                right = resample(right, int(stroke.shape[0] / 2))
                            else:
                                stroke = resample(stroke, (right.shape[0] * 2))

                            right_reflect = reflect_accross_center(right)
                            rights = connect(np.array(right), right_reflect)
                            rights = randomize_direction(rights)
                            added = 1

                            num_sym += rights.shape[0]
                            syms += 1
                            rights = parse.normalize_and_compute_deltas(rights)
                            right_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                                            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(rights.shape))),
                                            'ink': tf.train.Feature(float_list=tf.train.FloatList(value=rights.flatten()))}
                            right_example = tf.train.Example(features=tf.train.Features(feature=right_feature))
                            writer.write(right_example.SerializeToString())

                    if added:
                        num_not += stroke.shape[0]
                        nots += 1
                        stroke = parse.normalize_and_compute_deltas(stroke)
                        stroke_feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(stroke.shape))),
                                        'ink': tf.train.Feature(float_list=tf.train.FloatList(value=stroke.flatten()))}
                        stroke_example = tf.train.Example(features=tf.train.Features(feature=stroke_feature))
                        writer.write(stroke_example.SerializeToString())

    writer.close()
    print(syms)
    print(nots)
    print(num_sym / syms)
    print(num_not / nots)

def main():
    write_ink_record(["dataset/json/full-simplified-squiggle.ndjson"], 'dataset/reflected-eval', 0, 10000)
    write_ink_record(["dataset/json/full-simplified-squiggle.ndjson"], 'dataset/reflected-train', 10000, 100000)
    #write_ink_record(["dataset/json/full-simplified-squiggle.ndjson"], 'dataset/test', 0, 100)


if __name__ == '__main__':
    main()
