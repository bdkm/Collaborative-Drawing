import tensorflow as tf
import functools
import logging
import numpy as np

""" Creates an input function """
def get_input_fn(pattern, mode, batch_size):
    """ Parse a single record """
    def parse(single_file, mode):
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
    dataset = dataset.map(functools.partial(parse, mode=mode), num_parallel_calls=10)
    # Prefetch for performance
    dataset = dataset.prefetch(10000)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Shuffle data
        dataset = dataset.shuffle(buffer_size=1000000)
    # Pad inputs to make same length
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

def loader(filename):
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
    dataset = dataset.repeat()
    # Map parsing function over filenames
    return dataset.map(parse)


def loader2(strokes):
    labels = np.array(['class_index','shape','ink'])
    features = np.array([0,len(strokes),strokes])
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))

def predict_input_fn(dataset, batch_size, size):
    # Pad inputs to make same length
    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
    dataset = dataset.take(size)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

""" Defines the model for the network """
def my_model(features, labels, mode, params):
    """ Reshape inks to nx3, take first column of size and squeeze into row,
    squeeze labels into row """
    def get_input_tensors(features, labels):
        shapes = features['shape']
        # Takes height column of shapes
        lengths = tf.slice(shapes, begin=[0,0], size=[params.batch_size, 1])
        # Reshape into 1d vector
        lengths = tf.reshape(lengths,[params.batch_size])
        # Reshape ink into 8 x h x 3
        inks = tf.reshape(features['ink'], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inks, lengths, labels

    """ Convolutional layers """
    def conv_layers(inks, lengths):
        convolved = inks
        for i in range(len(params.num_conv)):
            convolved_input = convolved
            if i > 0 and params.dropout:
                convolved_input = tf.layers.dropout(
                    convolved_input,
                    rate=params.dropout,
                    training=(mode == tf.estimator.ModeKeys.TRAIN)
                )
            convolved = tf.layers.conv1d(
                convolved_input,
                filters=params.num_conv[i],
                kernel_size=params.conv_len[i],
                activation=None,
                strides=1,
                padding="same",
                name="conv1d_%d" % i)
        return convolved, lengths

    def rnn_layers(convolved, lengths):
        cell = tf.nn.rnn_cell.BasicLSTMCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lengths,
            dtype=tf.float32,
            scope="rnn_classification")

        mask = tf.tile(tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2), [1, 1, tf.shape(outputs)[2]])
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs

    def fc_layers(final_state):
        return tf.layers.dense(final_state, params.num_classes)

    inks, lengths, labels = get_input_tensors(features, labels)
    convolved, lengths = conv_layers(inks, lengths)
    final_state = rnn_layers(convolved, lengths)
    logits = fc_layers(final_state)
    print(logits)
    predictions = tf.argmax(logits, axis=1)

    """ Predictions """
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    """ Train and Evaluate """
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer="Adam",
        clip_gradients=params.gradient_clipping_norm,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            "logits": logits,
            "predictions": predictions},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={
            "accuracy": tf.metrics.accuracy(labels, predictions)
            })

def get_classifier(batch_size):
    config = tf.estimator.RunConfig(
        model_dir="models/shape_model",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        num_conv=[48, 64, 96],
        conv_len=[5, 5, 3],
        num_nodes=128,
        num_layers=3,
        num_classes=7,
        learning_rate=0.0001,
        gradient_clipping_norm=9.0,
        dropout=0.3)

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        config=config,
        params=params
    )

    return classifier

def predict(filename, size):
    classifier = get_classifier()
    predictions = classifier.predict(input_fn=lambda:predict_input_fn(filename, 8, size))
    return predictions

def main():
    # Show info when training
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)

    classifier = get_classifier()

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:get_input_fn("dataset/reflected-train.tfrecords", tf.estimator.ModeKeys.TRAIN, 8),
        max_steps=20000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:get_input_fn("dataset/reflected-eval.tfrecords", tf.estimator.ModeKeys.EVAL, 8)
    )
    #predict("08test-00000-of-00007.tfrecords")

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    """
    tensor = predict_input_fn('08test-00000-of-00007.tfrecords', 1)

    with tf.Session() as sess:
        for i in range(0,2):
            f, l = sess.run(tensor)
            i, l, ll = get_input_tensors(f, l)

            print(i)
            """


if __name__ == '__main__':
    main()
