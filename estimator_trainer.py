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
        # Initialise cell for each layer with num_nodes number of units
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        # Also initialise for backpropogation
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        # Add dropout if appropriate
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
        # Stack LSTM layers for forward and backpropogation
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lengths,
            dtype=tf.float32,
            scope="rnn_classification")

        # Zero out regions where sequences have no data (and so LSTM is just thinking)
        mask = tf.tile(tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2), [1, 1, tf.shape(outputs)[2]])
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        # Produce fixed length embedding by summing outputs of LSTM
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs

    def fc_layers(final_state):
        return tf.layers.dense(final_state, params.num_classes)

    inks, lengths, labels = get_input_tensors(features, labels)

    convolved, lengths = conv_layers(inks, lengths)
    final_state = rnn_layers(convolved, lengths)
    logits = fc_layers(final_state)

    predictions = tf.argmax(logits, axis=1)

    """ Predictions """
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    """ Train and Evaluate """
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    eval_metric_ops = {"accuracy": accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_classifier(batch_size):
    config = tf.estimator.RunConfig(
        model_dir="models/shape_model_qd",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        num_conv=[48,64,96], # Sizes of each convolutional layer
        conv_len=[5,5,3], # Kernel size of each convolutional layer
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
