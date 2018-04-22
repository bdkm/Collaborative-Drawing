# Load a json file
# Preprocess in usual way
# Split canvas in arbitrary direction, d
# Reflect points to generate two records
# Save with class as d


import tensorflow as tf
import numpy as np
import generator

"""
Loads quickdraw json file, parses contents and exports tfrecord holding
individual strokes
"""

def write_ink_record(dest_name, start, end):
    filename = "%s.tfrecords" % (dest_name)
    writer = tf.python_io.TFRecordWriter(filename)
    num_sym = 0
    syms = 0
    num_not = 0
    nots = 0

    for i in range(start,end):
        if i % 1000 == 0:
            print(i)

        ink, class_index = generator.generate()


        feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index])),
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(ink.shape))),
                        'ink': tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()

def main():
    write_ink_record('dataset/reflected-eval', 0, 10000)
    write_ink_record('dataset/reflected-train', 10000, 100000)


if __name__ == '__main__':
    main()
