from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

def rand_laplace(shape=[1]):
    exp1=tf.random_gamma(shape=shape,alpha=1,beta=1)
    exp2=tf.random_gamma(shape=shape,alpha=1,beta=1)
    return exp1-exp2

def network_mnist(images,input_shape,y_dim,mode,noise_type, noise):
    # features=images,labels,mode=TEST or TRAIN
    # Input Layer
    input_layer = tf.reshape(images, [-1]+input_shape)

    input_layer = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[0]>0),
                        lambda: input_layer+noise[0]*tf.random_normal(shape=tf.shape(input_layer)),
                        lambda: input_layer)
    input_layer = tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[0]>0),
                        lambda: input_layer+noise[0]/np.sqrt(2)*rand_laplace(shape=tf.shape(input_layer)),
                        lambda: input_layer)
    # if noise_type=='Gauss':
    #     input_layer=input_layer+noise[0]*tf.random_normal(shape=tf.shape(input_layer))
    # if noise_type=='Laplace':
    #     input_layer=input_layer+noise[0]*rand_laplace(shape=tf.shape(input_layer))

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    conv1 = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[1]>0),
                        lambda: conv1+noise[1]*tf.random_normal(shape=tf.shape(conv1)),
                        lambda: conv1)
    conv1 = tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[1]>0),
                        lambda: conv1+noise[1]/np.sqrt(2)*rand_laplace(shape=tf.shape(conv1)),
                        lambda: conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool1 = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[2]>0),
                        lambda: pool1+noise[2]*tf.random_normal(shape=tf.shape(pool1)),
                        lambda: pool1)
    pool1 = tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[2]>0),
                        lambda: pool1+noise[2]/np.sqrt(2)*rand_laplace(shape=tf.shape(pool1)),
                        lambda: pool1)

    # if noise[2].eval():
    #     pool1 += noise[2] * tf.random_normal(shape=tf.shape(pool1))
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    conv2 = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[3]>0),
                        lambda: conv2+noise[3]*tf.random_normal(shape=tf.shape(conv2)),
                        lambda: conv2)
    conv2 = tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[3]>0),
                        lambda: conv2+noise[3]/np.sqrt(2)*rand_laplace(shape=tf.shape(conv2)),
                        lambda: conv2)


    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2 = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[4]>0),
                        lambda: pool2+noise[4]*tf.random_normal(shape=tf.shape(pool2)),
                        lambda: pool2)
    pool2= tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[4]>0),
                        lambda: pool2+noise[4]/np.sqrt(2)*rand_laplace(shape=tf.shape(pool2)),
                        lambda: pool2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.relu, name='embedding')

    dense = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[5]>0),
                        lambda: dense+noise[5]*tf.random_normal(shape=tf.shape(dense)),
                        lambda: dense)
    dense =  tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[5]>0),
                        lambda: dense+noise[5]/np.sqrt(2)*rand_laplace(shape=tf.shape(dense)),
                        lambda: dense)
    dropout = tf.cond(tf.equal(mode,'TRAIN'),
                        lambda:tf.layers.dropout(inputs=dense, rate=0.4),
                        lambda:tf.layers.dropout(inputs=dense, rate=1))
    # if mode == "TRAIN":
    #     dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    # else:
    #     dropout = tf.layers.dropout(inputs=dense, rate=1)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=y_dim,name='logits')


    logits = tf.cond(tf.logical_and(tf.equal(noise_type,'Gauss'),noise[6]>0),
                        lambda: logits+noise[6]*tf.random_normal(shape=tf.shape(logits)),
                        lambda: logits)
    logits= tf.cond(tf.logical_and(tf.equal(noise_type,'Laplace'),noise[6]>0),
                        lambda: logits+noise[6]/np.sqrt(2)*rand_laplace(shape=tf.shape(logits)),
                        lambda: logits)
    #Returns logits and representer
    return logits

def cross_entropy_loss(logits, labels):
    """Cross entropy loss
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
    loss
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def representer_grad_loss(grad_representer):
    """control of the gradient of the representer
    Args:
    representer
    Returns:
    tr(grad(rep).T*grad(rep))
    """
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(grad_representer, grad_representer),axis=[1,2,3,4]))

def accuracy(y_pred,y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
