import numpy as np
from model import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from networks import *
#from model_adv import *
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.model import Model,CallableModelWrapper
#from foolbox.models import TensorFlowModel

train_images = np.load('/linkhome/rech/grpgen/urz85ee/DeepRob/MNIST/train_images.npy')
train_labels = np.load('/linkhome/rech/grpgen/urz85ee/DeepRob/MNIST/train_labels.npy')
a = train_labels
b = np.zeros((len(a), 10))
b[np.arange(len(a)), a] = 1
train_labels = b

test_images = np.load('/linkhome/rech/grpgen/urz85ee/DeepRob/MNIST/test_images.npy')
test_labels = np.load('/linkhome/rech/grpgen/urz85ee/DeepRob/MNIST/test_labels.npy')



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

run_config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list='1'))
run_config.gpu_options.allow_growth = True

K=100
range_noise=[]
range_noise+=[np.linspace(0,0.8,K)]
range_noise+=[np.linspace(0,0.2,K)]
range_noise+=[np.linspace(0,0.6,K)]
range_noise+=[np.linspace(0,0.5,K)]
range_noise+=[np.linspace(0,0.7,K)]
range_noise+=[np.linspace(0,10,K)]
range_noise+=[np.linspace(0,10,K)]

for i in range(7):
    range_noise_i=range_noise[i]

    for k in range(K):
        noise=np.zeros(1000)
        noise[i]=range_noise_i[k]


        tf.reset_default_graph()

        with tf.Session(config=run_config) as sess:
            cnn = CNN(sess,
                        y_dim=10,
                        batch_size=64,
                        epoch=5,
                        learning_rate=0.002,
                        beta=.5,
                        model_name='CNNdpgauss'+str(i)+'noise'+str(k),
                        checkpoint_dir="/linkhome/rech/grpgen/urz85ee/DeepRob/checkpoint")

            cnn.train(train_images, train_labels,noise_type='Gauss',noise=noise)


#
# grad = tf.gradients(cnn.global_loss, cnn.inputs)[0]
# a=grad.eval({
#     cnn.inputs: train_images[:2].reshape(tuple([-1]+cnn.input_shape)),
#     cnn.labels: train_labels[:2],
#     cnn.mode: 'TEST'
# })
#
# #y_pred = np.argmax(dcgan.predict(test_images), axis=1)
# model=CallableModelWrapper(cnn.predict, output_layer='logits')
#
#
#
# fgm=FastGradientMethod(model,sess=sess)
#
# fgm_params = {'eps': 0.3,
#                'clip_min': 0.,
#                'clip_max': 1.}
#
# adv_x = fgm.generate((train_images[:2]), **fgm_params)
# preds_adv = model.get_probs(adv_x)


#
# model = TensorFlowModel(cnn.inputs,cnn.network,bounds=(0, 255))
#
# from foolbox.criteria import TargetClassProbability
#
# target_class = 9
# criterion = TargetClassProbability(target_class, p=0.99)
#
#
# from foolbox.attacks import FGSM
#
#
# attack=FGSM(model)
# image = train_images[0].reshape((28, 28, 1))
# label = np.argmax(model.predictions(image))
#
# adversarial = attack(image,label=label,epsilons=1,max_epsilon=0.03*255)
#
#
# import matplotlib.pyplot as plt
#
# plt.subplot(1, 3, 1)
# plt.imshow(image.reshape((28, 28)), cmap='gray',vmin=0, vmax=255)
# plt.gca().set_title(label)
#
# plt.subplot(1, 3, 2)
# plt.imshow(adversarial.reshape((28, 28)), cmap='gray',vmin=0, vmax=255)
# plt.gca().set_title(np.argmax(model.predictions(adversarial)))
#
# plt.subplot(1, 3, 3)
# plt.imshow((adversarial - image).reshape((28, 28)), cmap='gray',vmin=0, vmax=255)
#
#
# embed=tf.get_default_graph().get_tensor_by_name("embedding/Relu:0")
#
# pp=[-1]+ cnn.input_shape+[1]
# gradient_embedding = tf.concat([tf.reshape(
#                            tf.gradients(embed[:,i], cnn.inputs)[0],pp) for i in range(embed.shape[1])],axis=4)
# loss_2 = networks.representer_grad_loss(gradient_embedding)
# loss_2.eval({
#     cnn.inputs: train_images[:100].reshape(tuple([-1]+ cnn.input_shape)),
#     cnn.labels: train_labels[:100],
#     cnn.mode: "TEST"
# })
