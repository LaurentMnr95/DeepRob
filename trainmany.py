import os







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
    for k in range(K):
        noise=range_noise[i][k]
        os.system('python3 train.py --layer '+str(i)+
                                    '--noise'+str(noise)+
                                    '--modelname gauss'+str(i)+'sd'+str(k))
