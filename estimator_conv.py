import tensorflow as tf
import logging
import numpy as np
import input_functions as input

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        layers.append(inks)
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
            layers.append(convolved)
        return convolved, lengths

    def fc_layers(final_state):
        return tf.layers.dense(final_state, params.num_classes)

    layers = []
    inks, lengths, labels = get_input_tensors(features, labels)
    final_state, lengths = conv_layers(inks, lengths)

    mask = tf.sequence_mask(lengths, tf.shape(final_state)[1])
    mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, tf.shape(final_state)[2]])
    layers.append(mask)
    zero_outside = tf.where(mask, final_state, tf.zeros_like(final_state))
    layers.append(zero_outside)
    final_state = tf.reduce_sum(zero_outside, axis=1)
    layers.append(final_state)
    logits = fc_layers(final_state)
    layers.append(logits)

    predictions = tf.argmax(logits, axis=1)
    layers.append([predictions])

    """
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    """

    plot_activations(layers)

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


    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
      }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def plot_activations(layers):
    print("Layers")
    with tf.Session() as sess:
        activations = sess.run(layers[0])
        width = len(activations)
        height = len(layers)
        print(activations)
        for j in range(len(activations)):
            plt.subplot(height, width, j + 1)
            plt.imshow(activations[j])


    for i in range(1,2):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('models/conv_model/model.ckpt-100000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('models/conv_model/'))
            outputTensors = sess.run(layers[0])
            #new_saver.restore(sess, tf.train.latest_checkpoint('models/conv_model/'))
            #print(sess.run('conv1d_0/kernel:0'))
            #init_g = tf.global_variables_initializer()
            #sess.run(init_g)
            #sess.run(init_l)
    """
            activations = sess.run(layers[i], feed_dict={tf.placeholder(tf.float32):activations})
            print(activations)

            if i > 3:
                plt.subplot(height, 1, i + 1)
                plt.imshow(activations)
            else:
                for j in range(len(activations)):
                    plt.subplot(height, width, (i * width) + j + 1)
                    plt.imshow(activations[j])

    plt.subplots_adjust(hspace = .1)
    plt.subplots_adjust(wspace = .1)
    plt.show()
    """
    print("End layers")

def get_classifier(batch_size):
    config = tf.estimator.RunConfig(
        model_dir="models/conv_model",
        save_checkpoints_secs=300,
        save_summary_steps=100)

    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        num_conv=[4],
        conv_len=[3],
        num_nodes=128,
        num_layers=3,
        num_classes=2,
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
        input_fn=lambda:input.batch_dataset("dataset/conv-train-???.tfrecords", tf.estimator.ModeKeys.TRAIN, 8),
        max_steps=100000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:input.batch_dataset("dataset/conv-eval-???.tfrecords", tf.estimator.ModeKeys.EVAL, 8)
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    main()
