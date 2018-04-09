import numpy as np
import tensorflow as tf
import simple_estimator as se
import ink_parser as ip

def ink_dataset(ink):
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

        return np.vstack(ink)

    ink = array_rep_to_ink_rep(ink)
    ink = ip.normalize_and_compute_deltas(ink)
    shape = ink.shape
    class_index = 1

    dataset = tf.data.Dataset.from_tensor_slices(
       {"ink": tf.constant([ink.flatten().tolist()]),
        "shape": tf.constant([shape]),
        "class_index": tf.constant([[class_index]])})
    dataset = dataset.padded_batch(1, padded_shapes=dataset.output_shapes)
    labels = tf.data.Dataset.from_tensor_slices(tf.constant([[[class_index]]]))
    return dataset.make_one_shot_iterator().get_next(), labels.make_one_shot_iterator().get_next()

def tfrecord_dataset(filename):
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
        labels = parsed_features["class_index"]
        return parsed_features, labels
    # Load in filenames holding tfrecords
    dataset = tf.data.TFRecordDataset(filename)

    # Map parsing function over filenames
    dataset = dataset.map(parse, num_parallel_calls=10)
    dataset = dataset.padded_batch(1, padded_shapes=dataset.output_shapes)
    dataset = dataset.take(1)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

ink = [[0.1,0.1,0.4,0.1,0.6,0.4],[0.7,0.3,0.8,0.8]]
#dataset1 = ink_dataset(ink)
#dataset2 = tfrecord_dataset('dataset/camera.tfrecords')

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

def classify(ink):
    classifier = se.get_classifier(1)
    predictions = classifier.predict(input_fn=lambda:ink_dataset(ink))
    colors = []
    for p in predictions:
        colors.append(class_to_color(p))
    return colors
