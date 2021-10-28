from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter
import whatnet.data.datasets.mnist as data_reader

if __name__ == "__main__":
    mnist = data_reader.read_data_sets("../scripts/data")

    # 参数 转换函数，图片的宽高
    converter = CnnConverter(param_file_path="../config/converter_config_default.json")
    # converter = converter = RdConverter("exponential", 28, 28)
    # converter1 = RdConverter("exponential", 28, 28)
    # 设置teacher信号的位置
    converter.teacher = 70.
    kernel = [[0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 1]]
    converter.ac_kernel(kernel)
    print(converter.pro_width)
    print(converter.pro_height)

    net = CnnNetwork(converter)
    # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
    net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
    net.create_outputlayer(10)
    net.link_inputlayer_outputlayer()

    # print(net.nest_local_node(net.get_input_neuron()))
    # print(net.nest_local_node(net.get_output_neuron()))

    # start = time.time()
    net.train(mnist.train.data[0], mnist.train.target[0])
    # 训练
    # 参数 数据集 最多迭代次数 准确率阈值
    # net.train_all(mnist.train.data[:1], mnist.train.target[:1], iter_max=1, accuracy_rate=.90)

    # end = time.time()
    # print(end - start)
    # predict
    # accuracy = net.predict_all(mnist.test.data[:10], mnist.test.target[:10])
    # if net.synchronizer.rank == 0:
    #     print("Accuracy in Test Set : %f", accuracy)

    # 保存网络结构
    # net.save_network_from_file("./test.csv")

    # net.get_output_trace()
    # net.get_input_trace()
    net.get_input_spikes()
    # net.get_output_spikes()
    net.show()
