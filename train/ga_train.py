from ENNs.whatnet.evolution.ga import GA
from ENNs.whatnet.cnn_network import CnnNetwork
from ENNs.whatnet.data.cnn_converter import CnnConverter
import ENNs.whatnet.data.datasets.mnist as data_reader

if __name__ == "__main__":
    mnist = data_reader.read_data_sets("./scripts/data")

    # 参数 转换函数，图片的宽高
    converter = CnnConverter("power", 28, 28, convolution_win_height=8, convolution_win_width=8, max_pooling_win_width=2, max_pooling_win_height=2)
    net = CnnNetwork(converter)
    # 参数 ： 卷积池化后图片的宽高 * 卷积核个数
    net.create_inputlayer(converter.pro_width * converter.pro_height * 4)
    net.create_outputlayer(10)
    net.link_inputlayer_outputlayer()

    # 训练
    # 参数 数据集 最多迭代次数 准确率阈值
    net.train_all(mnist.train.data[:1000], mnist.train.target[:1000], iter_max=5, accuracy_rate=.95)

    # predict
    accuracy = net.predict_all(mnist.train.data[:1000], mnist.train.target[:1000])

    GA_item = GA(net, net.converter.get_kernels(), accuracy, popu_num=30, generation_num=20, cross_rat=0.7, mutation_rat=0.05)
    GA_item.run()
