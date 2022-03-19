import tensorflow as tf

a = tf.constant([1])
b = tf.constant([2])
c = tf.add(a, b, "sum_c")

help(tf.add)
with tf.Session() as sess:
    sess.run(c)
    print(c.op)