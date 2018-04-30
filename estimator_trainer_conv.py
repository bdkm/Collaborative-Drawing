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
        lengths = tf.squeeze(tf.slice(shapes, begin=[0,0], size=[params.batch_size, 1]))
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
            if i != 2:
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
            else:
                tf.layers.max_pooling1d(
                    convolved_input,
                    pool_size=[2],
                    strides=1,
                    padding='same',
                    name='pool1d'
                )
        return convolved, lengths

    def fc_layers(final_state):
        return tf.layers.dense(final_state, params.num_classes)


    inks, lengths, labels = get_input_tensors(features, labels)
    convolved, lengths = conv_layers(inks, lengths)
    outputs = tf.reduce_sum(convolved, axis=1)
    logits = fc_layers(outputs)
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
        model_dir="models/shape_model_cnn_cao",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        num_conv=[48,64,96], # Sizes of each convolutional layer
        conv_len=[5,3,0,3], # Kernel size of each convolutional layer
        num_nodes=128, # Number of LSTM nodes for each LSTM layer
        num_layers=3, # Number of LSTM layers
        num_classes=7, # Number of classes in final layer
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

    for i in range(1,51):
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda:input.batch_dataset("dataset/shape-train-???.tfrecords", tf.estimator.ModeKeys.TRAIN, 8),
            max_steps= 1000 * i
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda:input.batch_dataset("dataset/shape-eval-???.tfrecords", tf.estimator.ModeKeys.EVAL, 8)
        )
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    main()
