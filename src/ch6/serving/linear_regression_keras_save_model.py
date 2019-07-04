import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

x_train = np.random.uniform(-10, 10, 100).reshape([-1, 1])
y_train = x_train * 2.0 + 5 + np.random.normal(0, 0.1, size=[100, 1])

layer = Dense(1, input_shape=(1,))
model = Sequential([layer])

model.compile(optimizer='sgd',
              loss='mse')

model.fit(x_train, y_train, epochs=100, batch_size=10)

print(layer.get_weights())

K.set_learning_phase(0)  # all new operations will be in test mode from now on

# serialize the model and get its weights, for quick re-building
config = model.get_config()
weights = model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = Sequential.from_config(config)
new_model.set_weights(weights)

# save mode
model_path = "lr_keras/1"
if tf.gfile.Exists(model_path):
    tf.gfile.DeleteRecursively(model_path)
builder = tf.saved_model.builder.SavedModelBuilder(model_path)
tensor_info_x = tf.saved_model.utils.build_tensor_info(new_model.input)
tensor_info_y = tf.saved_model.utils.build_tensor_info(new_model.output)

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

sess.close()