import tensorflow as tf
import numpy as np

w = tf.get_variable("w", shape=[1, 1], initializer=tf.random_uniform_initializer(-1, 1))
b = tf.get_variable("b", initializer=tf.constant(0.0))

x_train = np.random.uniform(-10, 10, 100).reshape([-1, 1])
y_train = x_train * 2.0 + 5 + np.random.normal(0, 0.1, size=[100, 1])
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(10)
iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()

y_ = tf.matmul(x, w) + b
loss = tf.reduce_mean(tf.square((y - y_)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(iterator.initializer)
        while True:
            try:
                sess.run(train_step)
            except tf.errors.OutOfRangeError:
                break

    w_pred, b_pred = sess.run([w, b])
    print("w: {}, b: {}".format(w_pred, b_pred))

    # save mode
    model_path = "lr_model_ds/1"
    if tf.gfile.Exists(model_path):
        tf.gfile.DeleteRecursively(model_path)
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y_)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': tensor_info_x},
            outputs={'y': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict':
                prediction_signature
        })

    builder.save()