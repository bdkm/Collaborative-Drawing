import numpy as np
import tensorflow as tf


dataset = tf.data.Dataset.from_tensor_slices(
   {"ink": tf.constant([4,3]),
    "shape": tf.constant([4, 100,5])})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
