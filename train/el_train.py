from ENNs.whatnet.cnn_network import CnnNetwork
from ENNs.whatnet.data.cnn_converter import CnnConverter
import ENNs.whatnet.data.datasets.mnist as data_reader
import nest

if __name__ == "__main__":
    mnist = data_reader.read_data_sets("./scripts/data")

    train_data = []
    train_label = []
    for idx in range(1):
        for _ in range(100):
            train_data.append(mnist.train.data[mnist.train.target == idx][_])
        for _ in range(100):
            train_label.append(mnist.train.target[mnist.train.target == idx][_])

    # 参数 转换函数，图片的宽高
    converter = CnnConverter(param_file_path="./config/converter_config_no_convolution.json")
    # 设置teacher信号的位置
    converter.teacher = 75.
    print(converter.pro_width)
    print(converter.pro_height)
    print(converter.kernels_num)

    net_cluster = list()

    cnt = 0

    for data in train_data:
        is_reg = False
        cnt += 1
        print(cnt)
        if len(net_cluster) == 0:
            net = CnnNetwork(converter)
            # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
            net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
            net.create_outputlayer(1)
            net.link_inputlayer_outputlayer()

            net.train(data, 0)

            net.save_network_from_file("./record/el/net_" + str(len(net_cluster)) + ".csv")
            net_cluster.append("./record/el/net_" + str(len(net_cluster)) + ".csv")
            del net

        for network_p in net_cluster:
            net = CnnNetwork(converter)
            net.build_network_from_file(network_p)
            if net.test(data):
                is_reg = True
                del net
                break
            del net
        if not is_reg:
            net = CnnNetwork(converter)
            # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
            net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
            net.create_outputlayer(1)
            net.link_inputlayer_outputlayer()

            net.train(data, 0)

            net.save_network_from_file("./record/el/net_" + str(len(net_cluster)) + ".csv")
            net_cluster.append("./record/el/net_" + str(len(net_cluster)) + ".csv")
            del net

    print(len(net_cluster))
