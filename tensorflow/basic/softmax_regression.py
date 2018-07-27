import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

print("Shape of feature matrix:", mnist.train.images.shape)
print("Shape of labels matrix:", mnist.train.labels.shape)

NUM_FEATURES = 784
NUM_LABELS = 10
LEARNING_RATE = 0.05
BATCH_SIZE = 128
NUM_EPOCHS = 51000

train_dataset = mnist.train.images
train_labels = mnist.train.labels
test_dataset = mnist.test.images
test_labels = mnist.test.labels
valid_dataset = mnist.validation.images
valid_labels = mnist.validation.labels

# Inputs
train_data_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_FEATURES))
train_labels_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
valid_data = tf.constant(valid_dataset)
test_data = tf.constant(test_dataset)

# Variables
weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_LABELS]))
biases = tf.Variable(tf.zeros([NUM_LABELS]))

# Training computation
logits = tf.matmul(train_data_batch, weights) + biases

# Function softmax_cross_entropy_with_logits labels of shape: [BATCH_SIZE, NUM_LABELS], and logits
# is of the same shape
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_batch, logits=logits))

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(tf.matmul(valid_data, weights) + biases)
test_prediction = tf.nn.softmax(tf.matmul(test_data, weights) + biases)

def accuracy(predictions, labels):
    correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correct) / predictions.shape[0]
    return accu

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(NUM_EPOCHS):
        offset = np.random.randint(0, train_labels.shape[0] - BATCH_SIZE - 1)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

        feed_dict = {train_data_batch: batch_data, train_labels_batch: batch_labels}

        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Mini batch loss at step {0}: {1}".format(step, l))
            print("Mini batch accuracy: {:.1f}%".format(accuracy(predictions, valid_labels)))
            print("Validation accuracy: {:.1f}%".format(accuracy(valid_prediction.eval(), valid_labels)))

    print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), test_labels)))


