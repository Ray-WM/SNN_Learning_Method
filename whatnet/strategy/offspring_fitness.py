import whatnet.data.datasets.mnist as data_reader
from whatnet.cnn_network import CnnNetwork
from whatnet.data.cnn_converter import CnnConverter


def get_fitness(net, kernels, test_set_num=1000):
    # Get Data
    mnist = data_reader.read_data_sets("./scripts/data")
    # Change net kernels
    net.converter.change_kernal(kernels)
    # predict
    accuracy = net.predict_all(mnist.test.data[:test_set_num], mnist.test.target[:test_set_num])
    return accuracy


def get_fitness_gc(dna, kernel, train_set_num=500, test_set_num=1000):
    # Get Data
    mnist = data_reader.read_data_sets("./scripts/data")
    # Change net kernels
    converter = CnnConverter(param_file_path="./config/converter_config_default.json")
    converter.ac_kernel(kernel)
    # Create net
    net = CnnNetwork(converter)
    net.create_inputlayer(converter.pro_width * converter.pro_height * converter.kernels_num)
    net.create_outputlayer(10)
    net.link_inputlayer_outputlayer()
    # train
    net.train_all(mnist.train.data[:train_set_num], mnist.train.target[:train_set_num], iter_max=2, accuracy_rate=.95)
    # predict
    accuracy = net.predict_all(mnist.test.data[:test_set_num], mnist.test.target[:test_set_num], is_record_error_msg=False)
    # log result
    if net.synchronizer.rank == 0:
        print("{ DNA: ", dna, ",accuracy: ", accuracy, "}")

    del net
    return accuracy
