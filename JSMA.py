import numpy as np
import tensorflow as tf

def SaliencyMap(image,target_class,gamma_set,cnn,sess,grad):
    g=sess.run(grad,feed_dict={
                            cnn.inputs: image.reshape([-1]+cnn.input_shape),
                            cnn.mode: "TEST",
                            cnn.noise: np.zeros(3)
                        })

    g=gamma_set*g
    sum_g_not_target = np.sum(g[np.delete(range(10),target_class)],axis=0)
    g_target=g[target_class]


    return np.abs(sum_g_not_target)*g_target*(sum_g_not_target<=0)*(g_target>=0)


def JSMA(sess,cnn,grad,image,target_class,max_iter=1000,theta=1/255,distortion_max=np.inf):
    image=image.reshape([28,28,1])
    curr_label = np.argmax(cnn.predict(image))

    gamma_set = np.ones((1,28,28,1))

    perturbed=image.copy()
    i=0

    while curr_label!=target_class and i<max_iter:
        S=SaliencyMap(perturbed,target_class,gamma_set,cnn,sess,grad)

        idx=np.unravel_index(np.argmax(S,axis=None),S.shape)

        perturbed[idx[1],idx[2],idx[3]]+=theta
        if perturbed[idx[1],idx[2],idx[3]]<=0 or perturbed[idx[1],idx[2],idx[3]]>=1:
            perturbed[idx[1],idx[2],idx[3]]=1
            gamma_set[:,idx[1],idx[2],idx[3]] = 0
        i+=1
        curr_label=np.argmax(cnn.predict(perturbed))
    return perturbed
