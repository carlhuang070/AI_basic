from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 黄黄

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):

    mnist=input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# Create the model
    x = tf.placeholder(tf.float32, [None, 784])    
# a 2-D tensor of floating-point numbers
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

# Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
 #
 #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
 #                                 reduction_indices=[1]))
 # tf.reduce_sum adds the elements in the second dimension of y, 
 # due to the reduction_indices=[1] parameter.
 # tf.reduce_mean computes the mean over all the examples in the batch.
 #
 # can be numerically unstable.
 #
 # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
 # outputs of 'y', and then average across the batch.

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 # apply your choice of optimization algorithm to modify the variables and reduce the loss.

    sess = tf.InteractiveSession()
 # launch the model in an InteractiveSession
    tf.initialize_all_variables().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # use tf.equal to check if our prediction matches the truth
  # tf.argmax(y,1) is the label our model thinks is most likely for each input, 
  # while tf.argmax(y_,1) is the correct label.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # [True, False, True, True] would become [1,0,1,1] which would become 0.75.
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

