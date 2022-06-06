# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import numpy as np
from models.mobilenet import MobileNet

#net = MobileNet(20,'normal')
#print(0.01*np.log(sum(param.numel() for param in net.parameters())))

# for pruning
def acc_reward(net, acc, flops):
    #params = sum(param.numel() for param in net.parameters())
    return acc * 0.1 - 0.015*np.log(flops)


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)
