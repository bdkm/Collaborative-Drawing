import tensorflow as tf
import numpy as np
import sys
import ink_parser as parse

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

def write_ink_record(source_name, dest_name, start, end):
    i = 0
    with open(source_name) as f:
        lines = f.readlines()

        for i in range(start,end):
            print("Splitting record %d" % i)
            filename = "%s-%05d.tfrecords" % (dest_name, i)
            writer = tf.python_io.TFRecordWriter(filename)
            l = lines[i]

            class_name, ink, recognised = parse.parse_json(l)
            ink = parse.reshape_ink(ink)
            strokes = split_strokes(ink)

            for stroke in strokes:
                stroke = parse.normalize_and_compute_deltas(stroke)
                feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(stroke.shape))),
                            'ink': tf.train.Feature(float_list=tf.train.FloatList(value=stroke.flatten()))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        i = i + 1
        writer.close()

def main(argv):
    print(argv[0])
    print(argv[1])
    argc = len(argv)
    if argc < 2:
        print("Usage: split_records <source> <output>")
    else:
        write_ink_record(argv[0], argv[1], 0, 20)


if __name__ == '__main__':
    main(sys.argv[1:])
