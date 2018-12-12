
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from math import floor, ceil
import random
from invmcc import gen_mfcc, mfcc_to_wav
import sys
from process_signal import *


model = sys.argv[1]
# model_input = sys.argv[2]
#506
#571
model_input = "Data/kaggle_cuts/english215/brother.wav"


def create_input(model_input):
    context_size = 30
    data = []
    mfcc1, y1, sr1 = gen_mfcc(model_input)
    for frame in range(mfcc1.T.shape[0]):
        if frame < context_size:
            cut = mfcc1.T[:context_size+frame, :]
            cut = np.pad(cut, ((context_size*2+1-len(cut),0),(0,0)), mode = "constant")
        elif mfcc1.T.shape[0] - frame - 1 < context_size:
            cut = mfcc1.T[frame-context_size:, :]
            cut = np.pad(cut, ((0,context_size*2+1-len(cut)),(0,0)), mode = "constant")
        else:
            cut = mfcc1.T[frame-context_size:context_size+frame+1, :]
        cut = cut.reshape(1,cut.shape[0]*cut.shape[1])
        data.append(cut[0])
    return np.array(data), y1, sr1

xdata, y, sr = create_input(model_input)
# output_nm = "/Users/sspusapaty/Desktop/reconstructions"+'_test_final2.wav'
# mfcc_to_wav(gen_mfcc(model_input)[0], y, sr, output_nm, 40)
# new_sig = SigProc(output_nm).logMMSE
# librosa.output.write_wav("/Users/sspusapaty/Desktop/processed2.wav", new_sig, 22050, norm=False)


def multilayer_perceptron(x, weights, biases, keep_prob):
    #layer_1 = tf.add(tf.matmul(tf.cast(x, tf.float32), weights['h1']), biases['b1'])
    layer_1 = tf.add(tf.matmul(tf.cast(x, tf.float32), weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_1 = tf.add(tf.matmul(tf.cast(layer_1, tf.float32), weights['h2']), biases['b2'])
    layer_1 = tf.nn.tanh(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(tf.cast(layer_1, tf.float32), weights['out']) + biases['out']
    return out_layer 


# # In[45]:

n_hidden_1 = 100
learning_rate = 0.01
n_input = 1525
n_classes = 75


keep_prob = tf.placeholder("float")

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
#pauls_rec_e1000_lr0001_100_100_smth_itself

x = tf.placeholder("float", [None, n_input])

saver = tf.train.Saver()

with tf.Session() as sess:
    # for w in weights:
    #     weights[w].initializer.run()
    # for b in biases:
    #     biases[b].initializer.run()
    saver.restore(sess, model+'/checkpoints.ckpt')

    output = multilayer_perceptron(xdata, weights, biases, 1).eval()

def average_mfccs(mfccs):
    averaged_mfccs = []
    coefs = 25
    
    averaged_mfccs.append((mfccs[0][coefs:2*coefs] + mfccs[1][0:coefs]) / 2)
    
    for i in range(1, len(mfccs)-1):
        averaged_mfccs.append((mfccs[i-1][2*coefs:3*coefs] + mfccs[i][coefs:2*coefs] + mfccs[i+1][0:coefs]) / 3)
    
    averaged_mfccs.append((mfccs[len(mfccs) - 2][2*coefs:3*coefs] + mfccs[len(mfccs) - 1][coefs:2*coefs]) / 2)
    
    return np.array(averaged_mfccs)

output = average_mfccs(output)
print("*********", output.shape)

mfcc_to_wav(output.T, y, sr, "/Users/sspusapaty/Desktop/"+model+'_output_final.wav', 40)
new_sig = SigProc("/Users/sspusapaty/Desktop/"+model+'_output_final.wav').logMMSE
librosa.output.write_wav("/Users/sspusapaty/Desktop/processed3.wav", new_sig, 22050, norm=False)




