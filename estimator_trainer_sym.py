import tensorflow as tf
import logging
import numpy as np
import input_functions as input

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


    inks, _, labels = get_input_tensors(features, labels)
    # currently slicing off initial position [0,1,0], some files have no points
    sliced = tf.slice(inks, tf.constant([0,1,0]), tf.constant([-1,-1,2]))
    repeated = tf.tile(sliced, tf.constant([1,1,2]))
    #negated = sliced * tf.constant([-1,1], tf.float32)
    xs = tf.slice(sliced, tf.constant([0,0,0]), tf.constant([-1,-1,1]))
    ys = tf.slice(sliced, tf.constant([0,0,1]), tf.constant([-1,-1,1]))
    convx = tf.layers.conv1d(
        xs,
        filters=1,
        kernel_size=1,
        activation=None,
        strides=1,
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="conv1d_x")
    convy = tf.layers.conv1d(
        ys,
        filters=1,
        kernel_size=1,
        activation=None,
        strides=1,
        padding="same",
        reuse=tf.AUTO_REUSE,
        name="conv1d_y")

    final = tf.concat([sliced, convx, convy], axis=2)
    sum = tf.reduce_sum(final, [1,2])

    predictions = tf.cast(tf.greater(tf.constant(0.1, tf.float32), tf.abs(sum)), tf.int32)

    """ Predictions """
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.abs(tf.reduce_sum(sum))
    """ Train and Evaluate """
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
      }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_classifier(batch_size):
    config = tf.estimator.RunConfig(
        model_dir="models/reflected_model",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        num_conv=[48,64,96], # Sizes of each convolutional layer
        conv_len=[5,5,3], # Kernel size of each convolutional layer
        num_nodes=128, # Number of LSTM nodes for each LSTM layer
        num_layers=3, # Number of LSTM layers
        num_classes=2, # Number of classes in final layer
        learning_rate=0.0001,
        gradient_clipping_norm=9.0,
        dropout=0.3)

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        config=config,
        params=params
    )

    return classifier

def main():
    # Show info when training
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)

    classifier = get_classifier(8)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:input.batch_dataset("dataset/reflected-train.tfrecords", tf.estimator.ModeKeys.TRAIN, 8),
        max_steps=100000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:input.batch_dataset("dataset/reflected-eval.tfrecords", tf.estimator.ModeKeys.EVAL, 8)
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    main()
