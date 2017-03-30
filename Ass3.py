
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


"""Stochastic gradient Decent"""
batch_size = 512
nodes_1 = 2048 #90.7% on test set with 1024 nodes #91.7 with 2048nodes #92% - 3layers, l2,
nodes_2 = 1024
nodes_3 = 1024
beta = 0.005 #for l2 regularisation
dropout_prob = 1 #89% with 0.5 #87.5% with no dropout #90.7% with beta =0.005 #87.5% with both
learning_rate = 0.0005

#problem 2: overfitting
#train_dataset = train_dataset[:5*batch_size,:]
#train_labels = train_labels[:5*batch_size,:]

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size), name="train_dataset")
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name="train_labels")
  tf_valid_dataset = tf.constant(valid_dataset, name="valid_dataset")
  tf_test_dataset = tf.constant(test_dataset, name="test_dataset")

  # Variables.
  weights = {
    'h1': tf.Variable(tf.truncated_normal([image_size * image_size, nodes_1]), name="h1"),
    'h2': tf.Variable(tf.truncated_normal([nodes_1, nodes_2]), name="h2"),
    'h3': tf.Variable(tf.truncated_normal([nodes_2, nodes_3]), name="h3"),
    'out': tf.Variable(tf.truncated_normal([nodes_3, num_labels]), name="out")
  }
  biases = {
    'b1': tf.Variable(tf.zeros([nodes_1]), name="b1"),
    'b2': tf.Variable(tf.zeros([nodes_2]), name="b2"),
    'b3': tf.Variable(tf.zeros([nodes_3]), name="b3"),
    'out': tf.Variable(tf.zeros([num_labels]), name="out")
  }
  #control drop-out with this - want to make 0.5 in training and 1 during predicting
  keep_prob = tf.placeholder(tf.float32, name="dropout_prob")

  def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

  # Training computation.
  # Construct model
  logits = multilayer_perceptron(tf_train_dataset, weights, biases)

  # Define loss and optimizer
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  #logits = multilayer_perceptron(tf_train_dataset, weights, biases)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + beta * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']) + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])))

  #global_step = tf.Variable(0)  # count the number of steps taken.
  #tf_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=500, decay_rate=0.96)
  #optimizer = tf.train.GradientDescentOptimizer(tf_learning_rate).minimize(loss, global_step=global_step)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(multilayer_perceptron(tf_test_dataset, weights, biases))



num_steps = 6001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels
        , keep_prob : dropout_prob}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval({keep_prob:1}), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval({keep_prob:1}), test_labels))
