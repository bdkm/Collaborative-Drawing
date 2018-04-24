import tensorflow as tf
import numpy as np
import ink_parser as parse

"""
Loads quickdraw json file, parses contents and exports tfrecord
"""

""" Writes out tfrecord files from json """
def write_ink_record(source_names, dest_name, start, end):
    class_index = 0
    for source_name in source_names:
        print("Exporting: " + source_name)
        filename = "%s-%03d.tfrecords" % (dest_name, class_index)
        writer = tf.python_io.TFRecordWriter(filename)
        with open(source_name) as f:
            lines = f.readlines()

            for i in range(start,end):
                l = lines[i]

                class_name, ink, recognised = parse.parse_element(l)
                if not recognised:
                    continue
                if np.sum(ink.flatten()[2::3]) <= 1.0:
                    feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index])),
                               'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(ink.shape))),
                               'ink': tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten()))}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
        class_index = class_index + 1
        writer.close()

def main():
    json_names = [ "dataset/json/full-simplified-squiggle.ndjson"
                 , "dataset/json/full-simplified-circle.ndjson"
                 , "dataset/json/full-simplified-octagon.ndjson"
                 , "dataset/json/full-simplified-hexagon.ndjson"
                 , "dataset/json/full-simplified-square.ndjson"
                 , "dataset/json/full-simplified-triangle.ndjson"
                 , "dataset/json/full-simplified-line.ndjson" ]

    write_ink_record(json_names, 'shape-train', 0, 10000)
    write_ink_record(json_names, 'shape-eval', 10000, 11000)
    #write_ink_record(['dataset/json/full-simplified-line.ndjson', 'dataset/json/full-simplified-zigzag.ndjson'], 'dataset/conv-train', 0, 100000)
    #write_ink_record(['dataset/json/full-simplified-line.ndjson', 'dataset/json/full-simplified-zigzag.ndjson'], 'dataset/conv-eval', 100000, 110000)



if __name__ == '__main__':
    main()
