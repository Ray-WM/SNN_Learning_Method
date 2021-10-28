import numpy as np
import matplotlib.pyplot as plt
from whatnet.data.converter import ConverterBase
import whatnet.data.datasets.mnist as data_read


class RdConverter(ConverterBase):
    def __init__(self, converter_func, image_w, image_h):
        super().__init__(image_w * image_h)

        # training config
        self.bias = 50.
        self.inh_t = 35.
        self.teacher = 0.

        # converterFuc
        self.converterFunc = converter_func

    def data(self, data, step=8):
        datas = data.flatten()[self.seed].reshape(-1)
        spike_times = []
        for start in range(0, self.n, step):
            data_c = np.array(list(filter(lambda _: _ != 0., datas[start: start + step])), dtype=np.float)
            if self.converterFunc == "linear":
                data_c_value = 256. - data_c
            elif self.converterFunc == "inverse":
                data_c_value = 255. / data_c
            elif self.converterFunc == "power":
                data_c_value = np.array([self.mapping(xo, 255, 1, 0, -np.sqrt(255)) for xo in data_c]) ** 2 + 1
            elif self.converterFunc == "exponential":
                data_c_value = 2 ** np.array([self.mapping(xo, 255, 1, np.log2(255), np.log2(1)) for xo in (256. - data_c)])
            else:
                raise Exception("No this function!")
            data_c_value.sort()
            data_c_value = list(data_c_value + self.bias)
            for index, value in enumerate(data_c_value):
                data_c_value[index] = round(value, 1)
            spike_times.append(data_c_value)
        return spike_times

    def target(self, target, inh=list()):
        spike_times = [np.array([], dtype=np.float)] * 10
        if len(inh) != 0:
            for _ in set(inh):
                spike_times[_] = np.array([self.inh_t], dtype=np.float)
        spike_times[target] = np.array([self.teacher], dtype=np.float)
        return spike_times

    # value mapping
    @staticmethod
    def mapping(xo, vo_max, vo_min, vn_max, vn_min):
        return (vn_max - vn_min) / (vo_max - vo_min) * (xo - vo_min) + vn_min


# Unit Test
if __name__ == "__main__":
    pass
