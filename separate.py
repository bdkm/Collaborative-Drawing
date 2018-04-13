import tensorflow as tf
import functools
import ink_parser
import estimator_shape as est
import numpy as np
import quickdraw_functions as qd

"""
Loads a tfrecord and runs each tensor through estimator_shape, outputting a
list of colors to represent the predicted class.
"""

def get_input_tensors(features, labels):
    shapes = features['shape']
    lengths = tf.squeeze(tf.slice(shapes, begin=[0,0], size=[params.batch_size, 1]))
    inks = tf.reshape(features['ink'], [params.batch_size, -1, 3])
    if labels is not None:
        labels = tf.squeeze(labels)
    return inks, lengths, labels

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

def class_to_label(c):
    mapping = {
        0: 'squiggle', # squiggle
        1: 'circle', # circle
        2: 'octagon', # octagon
        3: 'hexagon', # hexagon
        4: 'square', # square
        5: 'triangle', # triangle
        6: 'line'  # line
    }
    return mapping[c]

def predict_input_fn(filename, batch_size):
    """ Parse a single record """
    def parse(single_file):
        # Define feature column structure
        features = {
            'ink': tf.VarLenFeature(dtype=tf.float32),
            'shape': tf.FixedLenFeature([2], dtype=tf.int64),
            'class_index': tf.FixedLenFeature([1], dtype=tf.int64)
        }
        # Parse single record
        parsed_features = tf.parse_single_example(single_file, features)
        # Convert sparse tensor to dense (apply shape of tensor)
        parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
        # Separate out labels
        labels = parsed_features["class_index"]
        return parsed_features, labels
    # Load in filenames holding tfrecords
    dataset = tf.data.TFRecordDataset([filename])
    #dataset = dataset.repeat()
    # Map parsing function over filenames
    dataset = dataset.map(parse)
    # Pad inputs to make same length
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    #dataset = dataset.take(5)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

def get_input_tensors(features, labels, batch_size):
    shapes = features['shape']
    lengths = tf.squeeze(tf.slice(shapes, begin=[0,0], size=[batch_size, 1]))
    inks = tf.reshape(features['ink'], [batch_size, -1, 3])
    if labels is not None:
        labels = tf.squeeze(labels)
    return inks, lengths, labels

def separate(filename):
    """
    feature, label = predict_input_fn('dataset/camera.tfrecords', 2)
    with tf.Session() as sess:
        for _ in range(7):
            f = sess.run(feature)
            l = sess.run(label)

            inks, lengths, labels = get_input_tensors(f, l, 2)

            inks = sess.run(inks)
            print(inks)

    """
    n = 0
    for record in tf.python_io.tf_record_iterator(filename):
     n += 1
    predictions = est.predict(filename, n)

    colors = []
    for p in predictions:
        colors.append(class_to_color(p))

    print(len(colors))
    return colors
