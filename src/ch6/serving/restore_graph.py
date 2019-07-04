import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('tf_save_variables/model.ckpt.meta')
    saver.restore(sess, "tf_save_variables/model.ckpt")
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in vars:
        print("{}={}".format(var.name, sess.run(var)))