import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/mc/PycharmProjects/TestTensorflow/test/MNIST_data", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


h1_num=1000
h2_num=500

# 输入数据
x = tf.placeholder(tf.float32, [None, 784])


# 正确的标签
labels = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


W_fc1 = weight_variable([784, h1_num])
b_fc1 = bias_variable([h1_num])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([h1_num, h2_num])
b_fc2 = bias_variable([h2_num])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([h2_num, 10])
b_fc3 = bias_variable([10])
logits=tf.matmul(h_fc2_drop, W_fc3) + b_fc3

# 定义损失函数
# 1. unstable版本，不要使用！
#y = tf.nn.softmax(logits)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y), [1]))
# 2. stable版本
#scaled_logits = logits - tf.reduce_max(logits)
#normalized_logits = scaled_logits - tf.reduce_logsumexp(scaled_logits)
#cross_entropy =-tf.reduce_mean(labels * normalized_logits)
# 3. 推荐版本
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# define training step and accuracy
train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# initialize the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size=100

best_accuracy = 0.0
total_steps=0
train_kp_prop=0.5
for epcho in range(100):
    for i in range(60000//batch_size):
        input_images, correct_predictions = mnist.train.next_batch(batch_size)
        _, loss=sess.run([train_step,cross_entropy], feed_dict={x: input_images, labels: correct_predictions, keep_prob: train_kp_prop})
        if(i==0):
            print("step %d, loss %g" %(total_steps, loss))
        total_steps += 1
    train_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.train.images, labels: mnist.train.labels, keep_prob: 1})
    print("step %d, training accuracy %g" % (total_steps, train_accuracy))
    # validate
    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images, labels: mnist.test.labels, keep_prob:1})

    print("step %d, test accuracy %g" % (total_steps, test_accuracy))


