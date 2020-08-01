import tensorflow as tf
from tensorflow.python.framework import graph_util


with tf.Session(graph=tf.Graph()) as sess:
    data = tf.placeholder(tf.float32, shape=(32, 64), name='data')
    index = tf.placeholder(tf.int32, shape=(128), name='bond_index')
    #index = tf.cast(index, dtype=tf.int32)
    output =  tf.gather(data, index, axis=1)

    print ('output:', output.name)
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
								output_node_names=['GatherV2'])
    print(constant_graph)
    with tf.gfile.FastGFile('./test.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
