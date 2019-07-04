import onnx
import numpy as np
from onnx_tf.backend import prepare
import tensorflow as tf

model_path = "lr.onnx"


onnx_model = onnx.load(model_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation

print(tf_rep.inputs)
print(tf_rep.outputs)
x_tensor = tf_rep.tensor_dict[tf_rep.inputs[0]]
y_tensor = tf_rep.tensor_dict[tf_rep.outputs[0]]

print(x_tensor)
print(y_tensor)
x = np.array([[1.0]], dtype=np.float32)
y = tf_rep.run(x)
print(y)
with tf.Session(graph=tf_rep.graph) as sess:
    # save mode
    model_path = "pytorch-to-tf/1"
    if tf.gfile.Exists(model_path):
        tf.gfile.DeleteRecursively(model_path)
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x_tensor)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y_tensor)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x},
        outputs={'y': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict':
                prediction_signature
        })

    builder.save()
