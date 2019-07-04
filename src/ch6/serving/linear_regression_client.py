from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def do_inference():
    host, port = "localhost:9000".split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    x_test = np.random.uniform(-100, 100, [10, 1]).astype(np.float32)
    for i in range(x_test.shape[0]):
        x = x_test[i:i + 1]
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'lr'
        request.model_spec.signature_name = 'predict'

        request.inputs['x'].CopyFrom(
            tf.contrib.util.make_tensor_proto(x, shape=[1, 1]))
        result_future = stub.Predict.future(request, 5.0)

        response = np.array(
            result_future.result().outputs['y'].float_val)
        print("x: {}, y: {}", x, response)


def main(_):
    do_inference()


if __name__ == '__main__':
    tf.app.run()