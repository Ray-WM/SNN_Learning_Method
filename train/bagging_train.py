from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter
import whatnet.data.datasets.mnist as data_reader
import numpy as np


def vote(result, target_list):
    r = np.array(result)
    r = r.T
    vote_result = []

    # Get vote result
    for _ in r:
        b = _.tolist()
        vote_result.append(max(set(b), key=b.count))

    count = 0

    for vote_r, target in zip(vote_result, target_list):
        if vote_r == target:
            count += 1

    return count / len(target_list)


def save_result(path, result):
    re_scv = np.array(result)
    np.savetxt(path, re_scv, fmt="%d")


def read_result(path):
    re_scv = np.loadtxt(path, dtype=np.int)
    return re_scv


if __name__ == "__main__":

    disconnect_w = 1.  # adjust parameter

    training_set_num = 1000
    testing_set_num = 10000
    classifier_num = 15

    mnist = data_reader.read_data_sets("./scripts/data")

    # 参数 转换函数，图片的宽高
    converter = CnnConverter(param_file_path="./config/converter_config_default.json")
    # 设置teacher信号的位置
    converter.teacher = 75.
    print(converter.pro_width)
    print(converter.pro_height)

    is_fail = False
    all_result = []
    rest_synapse = []

    # train
    for _ in range(classifier_num):
        training_set, target_set = mnist.train.data[_ * training_set_num:_ * training_set_num + training_set_num], mnist.train.target[_ * training_set_num:_ * training_set_num + training_set_num]

        net = CnnNetwork(converter)
        net.set_disconnect_rate(disconnect_w)
        # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
        net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
        net.create_outputlayer(10)
        net.link_inputlayer_outputlayer()
        accuracy, index_l = net.train_all_with_result(training_set, target_set, iter_max=2, accuracy_rate=.95)
        all_result.append(net.predict_all_with_result(mnist.test.data[:testing_set_num], mnist.test.target[:testing_set_num]))
        rest_synapse.append(net.get_rest_connections_num())
        # save_result("./record/b_result.txt", all_result)

        del net

    net = CnnNetwork(converter)
    if net.synchronizer.rank == 0:
        if not is_fail:
            accuracy = vote(all_result, mnist.test.target[:testing_set_num])
            rest_synapse_rate = sum(rest_synapse) / (converter.pro_width * converter.pro_height * converter.kernels_num * 10 * classifier_num)
            print("Boosting result : ", accuracy)
            print("rest_synapse_rate : ", rest_synapse_rate)
        else:
            print("Boosting Fail")
