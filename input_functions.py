import numpy as np
import tensorflow as tf
import ink_parser as ip

"""A selection of input functions for estimator_shape"""

def pad(list,length):
    list += [0 for _ in range(length - len(list))]
    return list

def ink_dataset(inks, batch_size):
    """Converts [[x1,y1,...],...] to [[x1,y1,s1],...]"""
    def array_rep_to_ink_rep(ink):
        def reshape_2d(list):
            list = np.array(list)
            return list.reshape(-1,2)

        ink = [reshape_2d(stroke) for stroke in ink]

        def stroke_delimiter(height):
            zeros = np.zeros((height,1))
            zeros[-1] = 1
            return zeros

        ink = [np.hstack((stroke,stroke_delimiter(stroke.shape[0]))) for stroke in ink]
        return ink

    inks = array_rep_to_ink_rep(inks)
    inks = [ip.normalize_and_compute_deltas(ink) for ink in inks]
    shapes = [ink.shape for ink in inks]
    inks = [ink.flatten().tolist() for ink in inks]
    class_indexes = [[1] for ink in inks]

    max_len = len(max(inks,key=len))
    inks = [pad(ink, max_len) for ink in inks]

    dataset = tf.data.Dataset.from_tensor_slices(
       {"ink": tf.constant(inks),
        "shape": tf.constant(shapes),
        "class": tf.constant(class_indexes)})
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    labels = tf.data.Dataset.from_tensor_slices(tf.constant([class_indexes]))
    return dataset.make_one_shot_iterator().get_next(), labels.make_one_shot_iterator().get_next()

""" Parse a single record """
def parse_record(single_file):
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
    labels = parsed_features["class_index"]
    return parsed_features, labels

def tfrecord_dataset(filename, batch_size):
    # Load in filenames holding tfrecords
    dataset = tf.data.TFRecordDataset(filename)
    # Map parsing function over filenames
    dataset = dataset.map(parse_record, num_parallel_calls=10)
    #dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    dataset = dataset.take(2)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


""" Creates an input function """
def batch_dataset(pattern, mode, batch_size):
    # Load in filenames holding tfrecords
    dataset = tf.data.TFRecordDataset.list_files(pattern)
    # Shuffle filenames
    dataset = dataset.shuffle(buffer_size=10)
    # Create indefinite repeats of filenames
    dataset = dataset.repeat()
    # Map function to create record dataset and interleave
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)
    # Map parsing function over filenames
    dataset = dataset.map(parse_record, num_parallel_calls=10)
    # Prefetch for performance
    dataset = dataset.prefetch(10000)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Shuffle data
        dataset = dataset.shuffle(buffer_size=1000000)
    # Pad inputs to make same length
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels
