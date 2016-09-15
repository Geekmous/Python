#!/usr/bin/env python
from scipy.misc import imread
import scipy
import numpy as np
from scipy import io
import theano.tensor as T
import lasagne
import theano
import time
import cPickle as pickle
def loadData() :
    data1 = scipy.io.loadmat('data_batch_1.mat');
    data2 = scipy.io.loadmat('data_batch_2.mat');
    data3 = scipy.io.loadmat('data_batch_3.mat');
    data4 = scipy.io.loadmat('data_batch_4.mat');
    data5 = scipy.io.loadmat('data_batch_5.mat');
    test = scipy.io.loadmat('test_batch.mat');

    Xtrain = data1['data']

    ytrain = data1['labels']

    Xvalid = test['data'];
    yvalid = test['labels'];

    n = np.shape(Xtrain)[0];

    Xtrain = Xtrain.reshape((n,3,32,32))
    ytrain = ytrain.reshape((n))

    Xvalid = Xvalid.reshape((10000, 3, 32, 32))
    yvalid = yvalid.reshape((10000))
    return Xtrain, ytrain, Xvalid, yvalid

def build_mlp(input_var = None) :
    l_in = lasagne.layers.InputLayer((None, 3, 32, 32), input_var= input_var)
    l_drop  = lasagne.layers.DropoutLayer(l_in, p= .2)
    l_hid1 = lasagne.layers.DenseLayer(l_drop,num_units=800)
    l_drop2 = lasagne.layers.DropoutLayer(l_hid1, p= .2)
    l_hid2 = lasagne.layers.DenseLayer(l_drop2, num_units=800)
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units= 10, nonlinearity= lasagne.nonlinearities.softmax)
    return l_out;

def build_cnn(input_var = None):
    network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var = input_var);
    network = lasagne.layers.Conv2DLayer(network, num_filters = 32, filter_size=(5, 5));
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))
    network = lasagne.layers.Conv2DLayer(network, num_filters= 32, filter_size=(5, 5));
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DropoutLayer(network, p= .5)
    network = lasagne.layers.DenseLayer(network, num_units= 256);
    network = lasagne.layers.DropoutLayer(network, p=.5)
    network = lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax);
    return network;
Xtrain,ytrain, Xvalid, yvalid = loadData()

inputs = T.tensor4('inputs')
target = T.ivector('targets')
print "Building network...."
net_work = build_cnn(inputs)
print "finish...."
p = lasagne.layers.get_output(net_work)
loss = lasagne.objectives.categorical_crossentropy(p, target)
loss = loss.mean()
params = lasagne.layers.get_all_params(net_work, trainable= True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum = 0.9)
train_fn = theano.function([inputs, target], loss, updates=updates);

test_prediction = lasagne.layers.get_output(net_work);
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target);
test_loss = test_loss.mean();
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis = 1), target));

val_fn = theano.function([inputs, target], [test_loss, test_acc], updates=updates);

print "start Training..."
for epoch in range(300):
    train_err = 0;
    train_batches = 0;
    start_time = time.time()
    loss = train_fn(Xtrain, ytrain);
    end_time = time.time();
    val_loss, val_acc = val_fn(Xvalid, yvalid);

    print "epoch : ", epoch;
    print "time : ", end_time - start_time  ;
    print "loss : " , loss
    print "val_loss : ", val_loss
    print "val_acc : ", val_acc

with open("./layer.txt",'w') as output:
    layers = lasagne.layers.get_all_param_values(net_work)
    pickle.dump(layers, output);
with open("./layer.txt", 'r') as input:
    param = pickle.load(input);
    print param
