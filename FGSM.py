import numpy as np
import tensorflow as tf


def FGSM(sess,cnn,grad,image,epsilon):
    image=image.reshape([28,28,1])
    g=sess.run(grad,feed_dict={
                            cnn.inputs: image.reshape([-1]+cnn.input_shape),
                            cnn.mode: "TEST",
                            cnn.noise: np.zeros(3)
                        })
    c=np.argmax(cnn.predict(image))
    g=g.reshape([10,28,28,1])
    return image-epsilon*np.sign(g[c])
