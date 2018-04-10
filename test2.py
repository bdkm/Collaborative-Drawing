import tensorflow as tf
import test.ink_parser as ip

ip.normalize_and_compute_deltas([])

s1 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
s2 = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[5, 4])
print(s1)
"""
data = [[1,2,3,4,5,6],[7,8,9,10]]
shapes = [[3,2],[2,2]]
classes = [0,1]
dataset = tf.data.Dataset.from_tensor_slices(
   {"data": tf.constant(data),
    "shape": tf.constant(shapes),
    "class": tf.constant(classes)})

iterator = dataset.make_one_shot_iterator().get_next()


with tf.Session() as sess:
    x = sess.run(dataset)
    print(x)
"""
