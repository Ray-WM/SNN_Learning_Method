from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter
import whatnet.data.datasets.mnist as data_reader
from whatnet.data.rd_converter import RdConverter
from whatnet.data.converter import ConverterBase

if __name__ == "__main__":
    mnist = data_reader.read_data_sets("./scripts/data")

    # 参数 转换函数，图片的宽高
    # converter = CnnConverter(param_file_path="./config/converter_config_default.json")
    converter = CnnConverter(param_file_path="./config/converter_config_no_convolution.json")
    # 设置teacher信号的位置
    converter.teacher = 75.
    # print(converter.pro_width)
    # print(converter.pro_height)

    net = CnnNetwork(converter)
    # # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
    # net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
    net.create_inputlayer(28 * 28)
    net.create_outputlayer(10)
    net.link_inputlayer_outputlayer()

    # 训练
    # 参数 数据集 最多迭代次数 准确率阈值
    net.train_all(mnist.train.data[:1000], mnist.train.target[:1000], iter_max=4, accuracy_rate=.95)
    #
    # # predict
    accuracy = net.predict_all(mnist.test.data[:1000], mnist.test.target[:1000], record_file="./record/err_msg.csv")

    net.save_network_from_file("./record/test.csv")



    # net.train_all(mnist.train.data[:10], mnist.train.target[:10], iter_max=2, accuracy_rate=.95)

    # # predict
    # accuracy = net.predict_all(mnist.test.data[:10], mnist.test.target[:10], record_file="../record/err_msg.csv")
    #
    # net.save_network_from_file("../record/test.csv")
