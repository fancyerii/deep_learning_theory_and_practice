## Variable()和get_variable()的区别以及name_scope和variable_scope的区别

```
jupyter notebook
# 打开variable_vs_get_variable.ipynb
```


## 线性回归的例子

```
jupyter notebook
# 打开basic.ipynb
```

## dataset

```
jupyter notebook
# 打开dataset.ipynb
```

**注意：第8个cell会死循环，需要我们手动通过"kenerl->interrupt"停止。**

## Batch Normalization

```
jupyter notebook
# 打开batch-norm.ipynb
```

## 时间序列预测

```
jupyter notebook
# 打开time-series.ipynb
```

## 使用LSTM来识别MNIST图片

```
python lstm-mnist.py
```

## TensorBoard示例(使用MNIST)

首先训练：
```
python cnn_mnist.py
```

然后运行TensorBoard(不用等训练结束)：
```
deep_learning_theory_and_practice/src/ch6$ tensorboard --logdir mnist_convnet_model/
# 然后用浏览器打开命令行里出现的链接，可能类似于http://lili-Precision-7720:6006
```


## 使用内置的Estimator实现Iris数据分类

```
python premade_estimator.py
```

## tf.train.Saver


### 保存变量

```
cd serving
python save_variables.py
```

我们指定保存的路径是"tf_save_variables/model.ckpt"，最终会生成一个目录tf_save_variables，它包含如下内容：

```
$ tree tf_save_variables/
tf_save_variables/
├── checkpoint
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
└── model.ckpt.meta
```




### 使用 inspect_checkpoint 查看保存的变量

```
python view_ckpt.py
```

### 恢复 Graph 的定义和变量


```
python restore_graph.py
```


## Tensorflow Serving

### 线性回归的SavedModel

```
python linear_regression_save_model.py
```

SavedModel API把模型保存到lr_model目录：
```
$ tree lr_model
lr_model
└── 1
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

```

### 使用saved_model_cli查看保存的模型

```
$ saved_model_cli show --dir lr_model/1

The given SavedModel contains the following tag-sets:
serve
```

接着看serve这个tag_set：
```
$ saved_model_cli show --dir lr_model/1 --tag_set serve

The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "predict"
```

接着查看一个signature：
```
$ saved_model_cli show --dir lr_model/1 --tag_set serve --signature_def predict

The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: add:0
Method name is: tensorflow/serving/predict
```

### 安装ModelServer

请参考[官网文档](https://www.tensorflow.org/tfx/serving/setup)，不过作者更建议使用Docker来运行ModelServer，更多内容参考[在Docker中使用Tensorflow Serving](https://fancyerii.github.io/books/tfserving-docker/)。不过下面为了和原书保持一致，还是假设直接安装了ModelServer，如果读者使用Docker，那么请参考上网启动服务。

### 启动Model Server服务

```
(py3.6-env) lili@lili-Precision-7720:~/codes/deep_learning_theory_and_practice/src/ch6/serving$ tensorflow_model_server --port=9000 --model_name=lr --model_base_path=/home/lili/codes/deep_learning_theory_and_practice/src/ch6/serving/lr_model

....
最终应该输出类似"I tensorflow_serving/model_servers/main.cc:323] Running ModelServer at 0.0.0.0:9000 ..."
的信息，则说明启动成功。
```

**注意：上面的命令一定要把\-\-model_base_path设置成lr_model的绝对路径，请读者更加自己的情况修改。**



### 使用Python Client测试
注意：我们首先需要安装Python clinet:
```
pip install tensorflow-serving-api
```
然后再运行：
```
python linear_regression_client.py
```

### Java Client

首先需要把jar包安装，读者也可以自己打包，参考[这个文章](https://medium.com/@junwan01/a-java-client-for-tensorflow-serving-grpc-api-d37b5ad747aa)，这个过程相对麻烦，还需要Tensorflow源代码)。作者是用到比较老的1.5.0版本的Tensorflow和gRPC 1.4编译的，不过似乎ProtoBuffer和gRPC还是比较稳定的，在新的版本的Tensorflow Serving一般也是可以使用的。如果不行，那么只能读者自己编译了。当然如果输入数据不大，建议使用新版本Tensorflow Serving的REST接口。


```
$ cd tensor-serving-client-test
$ mvn install:install-file -Dfile=tensorflow-serving-java-client-tf-1.5.0-SNAPSHOT.jar -DgroupId=com.easemob.ai.robotapi \
-DartifactId=tensorflow-serving-java-client -Dversion=tf-1.5.0-SNAPSHOT -Dpackaging=jar
$ mvn package
```

### REST client

新版的Tensorflow Serving提供REST接口，这样客户端调用更加方便，如果输入或者输出需要传输的数据特别大，gRPC的方式可能有优势，但是REST接口简单易用便于调试。因为本书基于Tensorflow 1.6.0，因此没有介绍这部分内容，感兴趣的读者请参考[官方文档](https://www.tensorflow.org/tfx/serving/api_rest)。

### Dataset作为输入使用SavedModel API

```
python linear_regression_dataset_save_model.py
```

运行后模型会保存在lr_model_ds，如果要使用这个模型请参考前面的"启动Model Server服务"，需要修改"model_name"和"model_base_path"两个参数。

### Keras训练的模型使用SavedModel API保存

```
python linear_regression_keras_save_model.py
```

## 将PyTorch模型转为TensorFlow模型

### 在PyTorch中将模型保存为onnx格式

```
python pytorch-to-onnx.py
```

最终保存的模型为lr.onnx。

### TensorFlow加载onnx文件然后用SavedModel导出

首先需要安装onnx-tensorflow：
```
pip install onnx-tf
```

然后运行：
```
python onnx-to-tf.py
```


