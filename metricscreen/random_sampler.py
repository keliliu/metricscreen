import numpy as np


class RandomSampler:

    def __init__(self, n, batch_size, weight=None, buf=2000):
        self.counter = 0
        self.n = n
        self.batch_size = batch_size
        self.weight = weight
        self.buf = buf
        self.__get_samples()

    def __get_samples(self):
        self.samples = np.random.choice(self.n, size=self.batch_size*self.buf,\
                                        p=self.weight).reshape((self.buf, self.batch_size))

    def next(self):

        if self.counter >= self.buf:
            self.counter = 0
            self.__get_samples()

        ss = self.samples[self.counter, :]
        self.counter += 1

        return ss
