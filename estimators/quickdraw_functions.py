import tensorflow as tf

"""
Quickdraw dataset specific functions
"""

""" Minimal input function for a quickdraw tensor """
def get_quickdraw_tensor(filename):
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
        return parsed_features
    # Load in filenames holding tfrecords
    dataset = tf.data.TFRecordDataset([filename])
    # Map parsing function over filenames
    dataset = dataset.map(parse)
    # Prefetch for performance
    features = dataset.make_one_shot_iterator().get_next()
    return features
