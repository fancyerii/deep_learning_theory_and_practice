package com.easemob.ai.robotapi.tutorial.tfserving.client.test;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class LrClient {
	protected final static Logger logger = LoggerFactory.getLogger(LrClient.class);

    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    // Initialize gRPC client
    public LrClient(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        //System.out.println("Start the predict client");

        String host = "localhost";
        int port = 9000;

        if (args.length == 1) {
            String[] server_pair = args[0].split("=");
            if (!server_pair[0].equals("--server")) {
                System.out.println("you can only specify server address, no other args");
                return;
            }
            String[] server = server_pair[1].split(":");
            host = server[0];
            port = Integer.parseInt(server[1]);
        }

        String modelName = "lr";
        long modelVersion = 1;

        // Run predict client to send request
        LrClient client = new LrClient(host, port);

        try {
            client.do_predict(modelName, modelVersion);
        }
        finally {
            client.shutdown();
        }
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void do_predict(String modelName, long modelVersion) throws IOException {
    	Random rnd=new Random(1234);
    	
    	for(int i=0;i<10;i++) {
    		float x=rnd.nextFloat()*10; 
            TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();
 
            featuresTensorBuilder.addFloatVal(x);


            TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
            TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
            TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).addDim(featuresDim2).build();
            featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_FLOAT).setTensorShape(featuresShape);
            TensorProto featuresTensorProto = featuresTensorBuilder.build();

            // Generate gRPC request
            com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
            Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName)
                    .setSignatureName("predict")
                    .setVersion(version).build();
            Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec)
                    .putInputs("x", featuresTensorProto).build();

            // Request gRPC server
            try {
                Predict.PredictResponse response = blockingStub.predict(request);
                java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputsMap();
                TensorProto tp = outputs.get("y");
   
                logger.info("x: {}, y: {}", x, tp.getFloatVal(0));

            } catch (StatusRuntimeException e) {
                logger.warn("RPC failed: {0}", e.getStatus());
                return;
            }
        }
 

    }
}
