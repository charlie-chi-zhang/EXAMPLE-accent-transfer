
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from math import floor, ceil
import random
import sys
import os


# # In[43]:

xdata = np.load("pretrainWednesdaydata.npy")

ydata = np.load("pretrainWednesdaylabels.npy")

cost = []

if not os.path.exists("models"):
	os.makedirs("models")

print("******",sys.argv[0])

folder_to_store_model = sys.argv[1] # "model1"
print(folder_to_store_model)
directory = "models/" + folder_to_store_model

if not os.path.exists(directory):
	os.makedirs(directory)

train_size = 0.99

train_cnt = int(xdata.shape[0] * train_size)
x_test = xdata[train_cnt:,:]
y_test = ydata[train_cnt:,:]
x_train = xdata[0:train_cnt,:]
y_train = ydata[0:train_cnt,:]


# # In[44]:

def multilayer_perceptron(x, weights, biases, keep_prob):

    layer_1 = tf.add(tf.matmul(tf.cast(x, tf.float32), weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_1 = tf.add(tf.matmul(tf.cast(layer_1, tf.float32), weights['h2']), biases['b2'])
    layer_1 = tf.nn.tanh(layer_1)
    out_layer = tf.matmul(tf.cast(layer_1, tf.float32), weights['out']) + biases['out']
    return out_layer 


# # In[45]:

n_hidden_1 = 100
learning_rate = 0.0001
n_input = x_train.shape[1]
n_classes = y_train.shape[1]



# In[88]:
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")


# In[94]:

training_epochs = 1000
display_step = 50
batch_size = 16

x = tf.placeholder("float", [None,n_input])
y = tf.placeholder("float", [None, n_classes])


# In[95]:

#cost function
predictions = multilayer_perceptron(x, weights, biases, keep_prob)
mse = tf.losses.mean_squared_error(y,predictions)


# In[96]:

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)


# In[113]:

saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, mse], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob:0.8
                            })
            avg_cost += c / total_batch
        cost.append(avg_cost)
        if epoch % display_step == 0:
            name = directory + "/" + folder_to_store_model + "_" + str(epoch)
            if not os.path.exists(name):
                os.makedirs(name)
            np.save(name+"/cost_over_time.npy", np.array(cost))
            save_path = saver.save(sess,name+'/checkpoints.ckpt')
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))

np.save(directory+"/cost_over_time.npy", np.array(cost))


