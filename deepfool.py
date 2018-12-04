import numpy as np
import tensorflow as tf



def deepfool(sess,cnn,grad,image,overshoot=0.02,max_iter=50):

    image=image.reshape([28,28,1])
    f_image = np.array(cnn.predict(image)).flatten()
    label = np.argmax(f_image)
    num_classes= len(f_image)

    pert_image=image

    curr_label=label
    r_tot=0
    loop_i=0
    f=f_image
    wp=[0]*num_classes
    fp=[0]*num_classes
    norm_wp=[0]*num_classes

    while curr_label == label and loop_i<max_iter:
        g=sess.run(grad,feed_dict={
                            cnn.inputs: pert_image.reshape([-1]+cnn.input_shape),
                            cnn.mode: "TEST",
                            cnn.noise_type:'',
                            cnn.noise: np.zeros(1000)
                        })
        g=g.reshape((10,28,28,1))
        pert=[]

        for k in range(num_classes):
            wp[k] = g[k]-g[label]
            fp[k]=f[k]-f[label]
            norm_wp[k] = np.sqrt(np.sum(wp[k]**2))
            if norm_wp[k]==0:
                pert.append(np.inf)
            else:
                pert.append(np.abs(fp[k])/norm_wp[k])

        lt=np.argmin(pert)
        if norm_wp[lt]!=0:
            r_i=pert[lt]*wp[lt]/norm_wp[lt]
        else:
            r_i=0.001*np.random.normal(size=image.shape)
        r_tot+=r_i

        pert_image = image+(1+overshoot)*r_tot
        loop_i+=1

        f = np.array(cnn.predict(pert_image)).flatten()
        curr_label = np.argmax(f)

    r_tot=(1+overshoot)*r_tot
    return r_tot,loop_i,curr_label,pert_image
