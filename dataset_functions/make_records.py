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
        filename = "%s-%05d-of-%05d.tfrecords" % (dest_name, class_index, 7)
        writer = tf.python_io.TFRecordWriter(filename)
        with open(source_name) as f:
            lines = f.readlines()

            for i in range(start,end):
                l = lines[i]

                ink, class_name = parse.parse_element(l)

                feature = {'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_index])),
                           'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(ink.shape))),
                           'ink': tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten()))}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        class_index = class_index + 1
        writer.close()

def main():
    json_names = [ "../../dataset/full-simplified-squiggle.ndjson"
                 , "../../dataset/full-simplified-circle.ndjson"
                 , "../../dataset/full-simplified-octagon.ndjson"
                 , "../../dataset/full-simplified-hexagon.ndjson"
                 , "../../dataset/full-simplified-square.ndjson"
                 , "../../dataset/full-simplified-triangle.ndjson"
                 , "../../dataset/full-simplified-line.ndjson" ]

    #write_ink_record(json_names, 'train', 0, 10000)
    #write_ink_record(json_names, 'eval', 10000, 11000)
    write_ink_record(['dataset/json/full-simplified-camera.ndjson'], 'dataset/camera', 0, 20)


if __name__ == '__main__':
    main()
