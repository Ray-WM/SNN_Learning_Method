from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter
import whatnet.data.datasets.mnist as data_reader
import random
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


def alpha(err_p):
    return 0.5 * np.log((1 - err_p) / err_p)


def select(p_list):
    ran = random.random()
    i_list = np.arange(0, len(p_list))
    sum_p = 0
    for index, p in zip(i_list, p_list):
        sum_p += p
        if ran < sum_p:
            return index


def create_training_set(p_list, data_set):
    p_train_index = list()
    p_train_data = list()
    p_train_target = list()
    for _ in range(len(p_list)):
        i_p = select(p_list)
        p_train_index.append(i_p)
        p_train_data.append(data_set.train.data[i_p])
        p_train_target.append(data_set.train.target[i_p])
    return p_train_index, p_train_data, p_train_target


if __name__ == "__main__":

    training_set_num = 1000
    testing_set_num = 10000
    classifier_num = 10

    pro_list = [1 / training_set_num] * training_set_num

    mnist = data_reader.read_data_sets("./scripts/data")

    # 参数 转换函数，图片的宽高
    converter = CnnConverter(param_file_path="./config/converter_config_default.json")
    # 设置teacher信号的位置
    converter.teacher = 75.
    print(converter.pro_width)
    print(converter.pro_height)

    is_fail = False
    all_result = []

    # train
    for _ in range(classifier_num):
        if _ != 0:
            index_list, training_set, target_set = create_training_set(pro_list, mnist)
        else:
            index_list, training_set, target_set = np.arange(training_set_num).tolist(), mnist.train.data[:training_set_num], mnist.train.target[:training_set_num]

        net = CnnNetwork(converter)
        # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
        net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
        net.create_outputlayer(10)
        net.link_inputlayer_outputlayer()

        accuracy, index_l = net.train_all_with_result(training_set, target_set, iter_max=2, accuracy_rate=.95)
        all_result.append(net.predict_all_with_result(mnist.test.data[:testing_set_num], mnist.test.target[:testing_set_num]))

        if accuracy < 0.5:
            is_fail = True
            break
        # boosting
        alp = alpha(1 - accuracy)
        true_set = set()
        false_set = set()
        for i, flag in enumerate(index_l):
            if flag:
                true_set.add(index_list[i])
            else:
                false_set.add(index_list[i])

        for i in true_set:
            pro_list[i] *= (np.e ** (-alp))
        for i in false_set:
            pro_list[i] *= (np.e ** alp)

        pro_list /= sum(pro_list)

        del net

    if not is_fail:
        accuracy = vote(all_result, mnist.test.target[:testing_set_num])

        print("Boosting result : ", accuracy)
    else:
        print("Boosting Fail")
