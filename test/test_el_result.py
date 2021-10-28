from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter
import whatnet.data.datasets.mnist as data_reader
import pandas as pd

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

    df = pd.DataFrame()
    for _ in range(23):
        net = CnnNetwork(converter)
        net.build_network_from_file("./record/el/net_" + str(_) + ".csv")
        r_list = list()
        for data in train_data:
            r_list.append(net.test(data))
        df[_] = r_list
        del net

    net = CnnNetwork(converter)
    if net.synchronizer.rank == 0:
        df.to_csv("./record/el_0_result.csv")
