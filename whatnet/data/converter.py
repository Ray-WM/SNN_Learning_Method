import numpy as np
import random
from random import shuffle
import logging

class ConverterBase(object):
    
    def __init__(self, n):
        random.seed(2)
        self.n = n
        self.seed = list(range(n))
        shuffle(self.seed)
    
    def data(self, image, step=8):
        data = 256. - image.flatten()[self.seed].reshape(-1)
        spike_times = []
        for start in range(0, self.n, step):
            data[start:start+step].sort()
            spike_times.append(list(filter(lambda _: _ != 256., data[start:start+step])))
        return spike_times
    
    def target(self, target):
        spike_times = [np.array([], dtype=np.float)] * 10
        spike_times[target] = np.array(list(range(1, 260, 128)), dtype=np.float)
        return spike_times