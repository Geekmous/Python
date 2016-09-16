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
def loadData(batchSize = 500) :
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

    Xtrain = data1['data'].reshape((n,3,32,32))
    ytrain = data1['labels'].reshape((n))

    Xvalid = Xvalid.reshape((10000, 3, 32, 32))
    yvalid = yvalid.reshape((10000))

    for i in xrange(n / batchSize) :
        yield Xtrain[i * batchSize : (i + 1) * batchSize], ytrain[i * batchSize : (i + 1) * batchSize], Xvalid[i * batchSize : (i + 1) * batchSize], yvalid[i * batchSize : (i + 1) * batchSize]
    Xtrain = data2['data'].reshape((n, 3, 32, 32))
    ytrain = data2['labels'].reshape((n))
    for i in xrange(n / batchSize) :
        yield Xtrain[i * batchSize : (i + 1) * batchSize], ytrain[i * batchSize : (i + 1) * batchSize], Xvalid[i * batchSize : (i + 1) * batchSize], yvalid[i * batchSize : (i + 1) * batchSize]
    Xtrain = data3['data'].reshape((n, 3, 32, 32))
    ytrain = data3['labels'].reshape((n))
    for i in xrange(n / batchSize) :
        yield Xtrain[i * batchSize : (i + 1) * batchSize], ytrain[i * batchSize : (i + 1) * batchSize], Xvalid[i * batchSize : (i + 1) * batchSize], yvalid[i * batchSize : (i + 1) * batchSize]
    Xtrain = data4['data'].reshape((n, 3, 32, 32))
    ytrain = data4['labels'].reshape((n))
    for i in xrange(n / batchSize) :
        yield Xtrain[i * batchSize : (i + 1) * batchSize], ytrain[i * batchSize : (i + 1) * batchSize], Xvalid[i * batchSize : (i + 1) * batchSize], yvalid[i * batchSize : (i + 1) * batchSize]
    Xtrain = data5['data'].reshape((n, 3, 32, 32))
    ytrain = data5['labels'].reshape((n))
    for i in xrange(n / batchSize) :
        yield Xtrain[i * batchSize : (i + 1) * batchSize], ytrain[i * batchSize : (i + 1) * batchSize], Xvalid[i * batchSize : (i + 1) * batchSize], yvalid[i * batchSize : (i + 1) * batchSize]



def build_mlp(input_var = None) :

    l_in = lasagne.layers.InputLayer((None, 3, 32, 32), input_var= input_var)

    l_drop  = lasagne.layers.DropoutLayer(l_in, p= .2)
    l_hid1 = lasagne.layers.DenseLayer(l_drop,num_units=800)
    l_drop2 = lasagne.layers.DropoutLayer(l_hid1, p= .2)

    l_out = lasagne.layers.DenseLayer(l_drop2, num_units= 10, nonlinearity= lasagne.nonlinearities.softmax)
    return l_out;

def build_cnn(input_var = None):
    network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var = input_var);
    #first ConvLayer
    network = lasagne.layers.Conv2DLayer(network, num_filters = 5, filter_size=(5, 5),pad=2)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3,3), stride=2)
    #second Convlayer
    network = lasagne.layers.Conv2DLayer(network, num_filters= 5, filter_size=(5, 5), pad=2)
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3, 3), stride=2， mode='average_inc_pad')
    #thrid Convlayer
    network = lasagne.layers.Conv2DLayer(network, num_filters=5, filter_size=(5, 5), pad=2)
    network = lasagne.layers.Pool2DLayer(network, pool_size=(3, 3), stride=2, mode='average_inc_pad')


    network = lasagne.layers.DenseLayer(network, num_units= 64);

    network = lasagne.layers.DenseLayer(network, num_units=10, nonlinearity=lasagne.nonlinearities.softmax);
    return network;





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
for epoch in range(100):

    train_loss = 0
    valid_acc = 0
    valid_loss = 0
    start_time = time.time()
    i = 0
    for Xtrain, ytrain, Xvalid, yvalid in loadData(500):
        loss = train_fn(Xtrain, ytrain)
        val_loss, val_acc = val_fn(Xvalid, yvalid)
        valid_acc += val_acc
        valid_loss += val_loss
        train_loss += loss
        i += 1


    end_time = time.time()
    print "epoch : %s take %d" % epoch, (end_time - start_time)
    print "train_loss : ", train_loss
    print "valid_loss : ", valid_loss
    print "valid_acc : ", (valid_acc / i)

with open("./layer.txt",'w') as output:
    layers = lasagne.layers.get_all_param_values(net_work)
    pickle.dump(layers, output)
with open("./layer.txt", 'r') as input:
    param = pickle.load(input)
    print param
