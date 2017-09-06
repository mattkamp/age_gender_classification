#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: user
"""

from __future__ import print_function
import tensorflow as tf
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from sklearn.model_selection import train_test_split  


"""
MLP for gender
"""
classes = ['Male', 'Female']
num_classes = len(classes)

# batch size
batch_size = 100

# validation split
validation_size = .16

img_size = 128

num_channels= 3

img_size_flat = img_size * img_size * num_channels

train_path = '/home/user/Desktop/Python/dataGender'
test_path = '/home/user/Desktop/Python/dataGenderTest/'

data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.valid.labels)))


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features

n_input = 128 * 128 * 3 # MNIST data input (img shape: 100*100)
n_classes = 2 # MNIST total classes (0:male 1:female)

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


#Weigths and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

gr_cost = []
gr_tr_acc = []
gr_test_acc = []

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(data.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_test_batch, y_test_batch, _, test_cls_batch = data.valid.next_batch(batch_size)
            
            x_batch = x_batch.reshape(batch_size, img_size_flat)
            x_test_batch = x_test_batch.reshape(batch_size, img_size_flat)

            _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_true_batch})
            batch_acc = accuracy.eval({x: x_batch, y: y_true_batch})
            # Compute average loss
            avg_cost += c / total_batch
            avg_acc += batch_acc / total_batch
       
        # Display logs per epoch step
        if epoch % display_step == 0:
            test_acc = accuracy.eval({x: x_test_batch.reshape(len(x_test_batch), img_size_flat), y: y_test_batch})
            
            gr_cost.append(avg_cost)
            gr_tr_acc.append(avg_acc)
            gr_test_acc.append(test_acc)
             
            print(  
                
                "Epoch:",
                '%04d' % (epoch+1),
                "cost=",
                "{:.9f}".format(avg_cost),
                "average_train_accuracy=",
                "{:.6f}".format(avg_acc),
                "test_accuracy=",
                "{:.6f}".format(test_acc)
                
            
            )

        print("Optimization Finished!")


axis_x = range(1,training_epochs+1)

fig = plt.figure()
fig.suptitle('Age classification MLP- Average accuracy test/train', fontsize=14, fontweight='bold')
plt.plot(axis_x, gr_tr_acc, '--bo', label ='Traing')
plt.plot(axis_x, gr_test_acc, '--ro', label ='Test')
plt.legend()
ax = fig.add_subplot(111)
ax.set_ylabel('Accuracy Perc')
ax.set_xlabel('Taining Epochs ')
ax.set_title('Images of 50x50 pixels', fontsize=12)
#fig.savefig('/home/user/Desktop/Python/age_128x128.png')

fig = plt.figure()
fig.suptitle('Age classification MLP- Average cost', fontsize=14, fontweight='bold')
plt.plot(axis_x, gr_cost, '--bo')
plt.legend()
ax = fig.add_subplot(111)
ax.set_ylabel('Cost')
ax.set_xlabel('Taining Epochs ')
ax.set_title('Images of 50x50 pixels', fontsize=12)
#fig.savefig('/home/user/Desktop/Python/age_cost_128x128.png')
