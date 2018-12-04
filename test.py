import mnist

from model import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from networks import *
#from model_adv import *
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.model import Model,CallableModelWrapper
#from foolbox.models import TensorFlowModel

train_images = mnist.train_images()/255
train_labels = mnist.train_labels()
a = train_labels
b = np.zeros((len(a), 10))
b[np.arange(len(a)), a] = 1
train_labels = b

test_images = mnist.test_images()/255
test_labels = mnist.test_labels()



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

i=0
range_noise=np.linspace(0,0.5,5)
for n in range_noise:
    noise=np.zeros(1000)
    noise[i]=n


    tf.reset_default_graph()

    with tf.Session() as sess:
        cnn = CNN(sess,
                    y_dim=10,
                    batch_size=64,
                    epoch=20,
                    learning_rate=0.002,
                    beta=.5,
                    model_name='CNNdpgauss'+str(i)+'noise'+str(n),
                    checkpoint_dir="checkpoint")


        grad = tf.concat([tf.reshape(
            tf.gradients(cnn.network[:,i], cnn.inputs)[0], [1,-1,28,28,1]) for i in range(10)], axis=0)

        prediction_test=np.argmax(cnn.predict(test_images),axis=1)
        print(np.mean(prediction_test==test_labels))






test_labels
