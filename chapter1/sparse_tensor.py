import tensorflow as tf
import numpy as np

"""
indices: 不为0的元素索引
values: indices处的值
dense_shape：稀疏矩阵的shape
"""
st = tf.SparseTensor(indices=[[1, 1], [2, 2]], values=[10, 20], dense_shape=[3, 3])

with tf.Session() as sess:
    print(st.values)
    print(tf.sparse_tensor_to_dense(st))


x = tf.sparse_placeholder(tf.float32)

indices = np.array([[0, 1],
                    [0, 3],
                    [1, 2],
                    [1, 3]], dtype=np.int64)
values = np.array([2, 1, 1, 1], dtype=np.int64)
dense_shape = np.array([2, 4], dtype=np.int64)

with tf.Session() as sess:
    sparse_tensor = sess.run(x, feed_dict={
        x: tf.SparseTensorValue(indices, values, dense_shape)})
    print(sparse_tensor)
    tensor_value = tf.sparse_tensor_to_dense(sparse_tensor)
    print('tensor表示的稀疏矩阵:\n', sess.run(tensor_value))
