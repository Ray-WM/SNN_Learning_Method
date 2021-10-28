import numpy as np
import matplotlib.pyplot as plt
import json
from whatnet.data.converter import ConverterBase
import whatnet.data.datasets.mnist as data_read


class CnnConverter(ConverterBase):
    def __init__(self, param_file_path='../config/converter_config_default.json'):

        with open(param_file_path, 'r') as f:
            self.params = json.load(f)

        super().__init__(28 * 28)
        self.con_win_w = self.params['convolution_win_width']
        self.con_win_h = self.params['convolution_win_height']
        self.mp_win_w = self.params['max_pooling_win_width']
        self.mp_win_h = self.params['max_pooling_win_height']

        self.operation = self.params['operation']

        # whether use custom kernel
        self.__is_c_kernel = self.params['c_kernel']

        # convolution kernel
        self.kernel1 = []
        self.kernel2 = []
        self.kernel3 = []
        self.kernel4 = []

        # custom convolution kernel
        self.kernels = {}

        # initialize kernels
        self.__init_kernel(self.con_win_w, self.con_win_h)
        self.kernels = self.params['kernels']
        if self.__is_c_kernel:
            self.kernels_num = len(self.kernels.keys())
        else:
            self.kernels_num = 4

        # training config
        self.bias = 50.
        self.inh_t = 35.
        self.teacher = 75.

        # converterFuc
        self.converterFunc = self.params['converter_func']

        # image width height
        self.img_width = self.params['image_w']
        self.img_height = self.params['image_h']

        # pro_width pro_height
        self.pro_width = self.params['image_w']
        self.pro_height = self.params['image_h']
        self.__init_pro_w_h()

    def __init_kernel(self, con_w, con_h):
        for i in range(con_h):
            if i == int(con_h / 2):
                self.kernel1.append([1] * con_w)
            else:
                self.kernel1.append([0] * con_w)

        for i in range(con_h):
            r = [0] * con_w
            r[int(con_w / 2)] = 1
            self.kernel2.append(r)

        for i in range(con_h):
            r = [0] * con_w
            r[i] = 1
            self.kernel3.append(r)

        for i in range(con_h):
            r = [0] * con_w
            r[con_w - i - 1] = 1
            self.kernel4.append(r)

    def __init_pro_w_h(self):
        self.pro_width = self.img_width
        self.pro_height = self.img_height
        for op in self.operation:
            if op == 'convolution':
                self.pro_width = self.pro_width - self.con_win_w + 1
                self.pro_height = self.pro_height - self.con_win_h + 1
            elif op == 'pooling':
                self.pro_width = int(self.pro_width / self.mp_win_w)
                self.pro_height = int(self.pro_height / self.mp_win_h)
            else:
                raise "No such operation in cnn_converter. You may try {" + "convolution or pooling" + "}."

    def change_kernal(self, kernels):
        mgs = np.shape(kernels)
        self.kernel1 = np.array(kernels[0]).tolist()
        self.kernel2 = np.array(kernels[1]).tolist()
        self.kernel3 = np.array(kernels[2]).tolist()
        self.kernel4 = np.array(kernels[3]).tolist()

        self.con_win_w = mgs[1]
        self.con_win_h = mgs[2]
        self.__init_pro_w_h()

    def get_kernels(self):
        kernels = [] + self.kernel1 + self.kernel2 + self.kernel3 + self.kernel4
        return np.array(kernels).flatten().reshape(4, 8, 8)

    def ac_kernel(self, kernel):
        self.kernels['ga_kernel'] = np.array(kernel).tolist()
        self.kernels_num = 5

    # data converter
    # ------------------------------------------------------------------------------
    def data(self, data, step=8):
        spike_times = []
        if not self.__is_c_kernel:
            intermediate_result = self.pre_pro(data)
        else:
            intermediate_result = self.pre_pro_c(data)

        max_p = self.get_max_value(intermediate_result)

        for image_kernel in intermediate_result.values():
            image_coved = self.function_conver(self.converterFunc, image_kernel, max_p, 0) + self.bias
            c = self.add_spike(image_coved)
            spike_times = spike_times + c

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

    @staticmethod
    def get_max_value(intermediate_result):
        max_p = 0
        for img in intermediate_result.values():
            max_p = max(max_p, max(img.flatten()))
        return max_p

    @staticmethod
    def __display_distribution(x, y):
        plt.scatter(x, y, color="red")
        plt.show()

    def function_conver(self, converfunc, data, max_value, min_value):
        if converfunc == "exponent":
            image_mapped = self.mapping(data.flatten(), max_value, min_value, 0, -np.log2(100))
            return 0.5 ** np.array(image_mapped, dtype=np.float)
        elif converfunc == "power":
            image_mapped = self.mapping(data.flatten(), max_value, min_value, 0, -np.sqrt(99))
            return np.array(image_mapped, dtype=np.float) ** 2 + 1
        elif converfunc == "linear":
            image_mapped = self.mapping(data.flatten(), max_value, min_value, 99, 0)
            return 100 - np.array(image_mapped, dtype=np.float)
        elif converfunc == "inverse":
            image_mapped = self.mapping(data.flatten(), max_value, min_value, 100, 1)
            return 100 / np.array(image_mapped, dtype=np.float)
        else:
            raise Exception("No this Function!")

    def add_spike(self, image_coved):
        spikes = []
        for v in image_coved:
            ss = round(v, 1)
            if ss != 100. + self.bias:
                spikes.append([ss])
            else:
                spikes.append([])
        return spikes

    # Convolution and Max Pooling
    # --------------------------------------

    @staticmethod
    def convolution(patch, kernal):
        return sum((patch * np.array(kernal, dtype=np.float)).flatten())

    def convolution_layer(self, data, kernal):
        image_shape = np.shape(data)
        image_data = np.array(data, dtype=np.float).flatten().reshape(image_shape[0], image_shape[1])
        self.pro_width = image_shape[0] - self.con_win_w + 1
        self.pro_height = image_shape[1] - self.con_win_h + 1
        con_image_data = [self.convolution(np.array(image_data[r:r + self.con_win_w, c:c + self.con_win_h], dtype=np.float), kernal=kernal)
                          for r in range(image_shape[0] - self.con_win_w + 1)
                          for c in range(image_shape[1] - self.con_win_h + 1)]
        return np.array(con_image_data, dtype=np.float).reshape(self.pro_width, self.pro_height)

    @staticmethod
    def max_pooling(patch):
        ret_value = max(patch.flatten())
        if ret_value <= 100:
            return 0
        else:
            return max(patch.flatten())

    def max_pooling_layer(self, data):
        image_shape = np.shape(data)
        image_data = np.array(data, dtype=np.float).flatten().reshape(image_shape[0], image_shape[1])
        self.pro_width = int(image_shape[0] / self.mp_win_w)
        self.pro_height = int(image_shape[1] / self.mp_win_h)
        mp_image_data = [self.max_pooling(np.array(image_data[r:r + self.mp_win_w, c:c + self.mp_win_h], dtype=np.float))
                         for r in range(0, (self.pro_width - 1) * self.mp_win_w + 1, self.mp_win_w)
                         for c in range(0, (self.pro_height - 1) * self.mp_win_h + 1, self.mp_win_h)]
        return np.array(mp_image_data, dtype=np.float).reshape(self.pro_width, self.pro_height)

    def pre_pro(self, data):
        intermediate_result = {'k1': data, 'k2': data, 'k3': data, 'k4': data}
        for op in self.operation:
            if op == 'convolution':
                intermediate_result['k1'] = self.convolution_layer(intermediate_result['k1'], kernal=self.kernel1)
                intermediate_result['k2'] = self.convolution_layer(intermediate_result['k2'], kernal=self.kernel2)
                intermediate_result['k3'] = self.convolution_layer(intermediate_result['k3'], kernal=self.kernel3)
                intermediate_result['k4'] = self.convolution_layer(intermediate_result['k4'], kernal=self.kernel4)
            elif op == 'pooling':
                intermediate_result['k1'] = self.max_pooling_layer(intermediate_result['k1'])
                intermediate_result['k2'] = self.max_pooling_layer(intermediate_result['k2'])
                intermediate_result['k3'] = self.max_pooling_layer(intermediate_result['k3'])
                intermediate_result['k4'] = self.max_pooling_layer(intermediate_result['k4'])
        return intermediate_result

    def pre_pro_c(self, data):
        intermediate_result = {}

        for key in self.kernels.keys():
            intermediate_result[key] = data

        for op in self.operation:
            if op == 'convolution':
                for key in self.kernels.keys():
                    intermediate_result[key] = self.convolution_layer(intermediate_result[key], kernal=self.kernels[key])
            elif op == 'pooling':
                for key in self.kernels.keys():
                    intermediate_result[key] = self.max_pooling_layer(intermediate_result[key])

        return intermediate_result

    @staticmethod
    def draw_data(ori_image, new_image=221):
        plt.subplot(new_image)
        plt.imshow(ori_image)


# Unit Test
if __name__ == "__main__":
    mnist = data_read.read_data_sets("../../scripts/data")
    converter = CnnConverter(param_file_path='../../config/converter_config_default.json')

    image = mnist.train.data[1]
    print("Target %s" % mnist.train.target[1])
    print(converter.pro_width, converter.pro_height)

    pre = converter.pre_pro_c(image)
    num = 1
    for value in pre.values():
        converter.draw_data(value, 220 + num)
        num += 1
        if num > 4:
            num = 1
            plt.figure()

    plt.show()
