import numpy as np
import pandas as pd

import nest
import progressbar
from whatnet.data.cnn_converter import CnnConverter
from whatnet.networkbase import NetworkBase
from whatnet.common.mpi_sync import Synchronizer

import nest.raster_plot
import nest.voltage_trace

import pylab

"""Network对nest进行封装，Network模块外不直接调用nest模块

负责神经元、突触和其他device的维护。神经元和突触的创建由外部调用接口完成
对涉及的模型参数由统一的函数（__nest__init）管理

需要支持的功能：
    1. 神经元的创建与销毁
    2. 突触的创建与销毁
    3. 设定输入/输出神经元
    4. 连接输入/输出神经元与信号发生器
    5. 提供设定输入信号的接口
    6. 执行网络功能：学习/预测
    7. 提供读取输出神经元数据功能
"""


class CnnNetwork(NetworkBase):
    def __init__(self, converter: CnnConverter):
        self.__nest_init()

        self.__layer = {}
        self.__predictLayer = {}

        self.__generator = {}
        self.__predictGenerator = None

        # noise init
        self.__noise_generator = {}
        self.__predictNoiseGenerator = None


        self.__teacher = None
        self.__detector = {}
        self.__multimeter = {}

        self.__duration = 500.
        self.__predict_duration = 500.
        self.__generator_weight = 200.
        self.converter = converter

        self.synchronizer = Synchronizer()

        self.progressbar = progressbar.ProgressBar()

        self.__disconnect_rate = 1.0

    def set_disconnect_rate(self, rate):
        self.__disconnect_rate = rate

    def __nest_init(self):
        nest.ResetKernel()
        # nest.SetKernelStatus({"local_num_threads": 4})
        nest.set_verbosity('M_ERROR')
        nest.CopyModel("iaf_psc_alpha", "input_neuron", params={"V_th": -68., "t_ref": 20.})
        nest.CopyModel("iaf_psc_alpha", "output_neuron", params={"V_th": -67., "t_ref": 300.})
        nest.CopyModel("iaf_psc_alpha", "wn_neuron")

    # Create Network
    # -------------------------------------------
    def create_neuron(self, *args, **kwargs):
        return nest.Create("wn_neuron", *args, **kwargs)

    @staticmethod
    def __create_output_neuron(*args, **kwargs):
        return nest.Create("output_neuron", *args, **kwargs)

    @staticmethod
    def __create_input_neuron(*args, **kwargs):
        return nest.Create("input_neuron", *args, **kwargs)

    def create_inputlayer(self, *args, **kwargs):
        self.__layer['inputLayer'] = self.__create_input_neuron(*args, **kwargs)
        self.__predictLayer['inputLayer'] = self.__create_input_neuron(*args, **kwargs)
        self.__mark_input()
        return self.__layer['inputLayer']

    def create_outputlayer(self, *args, **kwargs):
        self.__layer['outputLayer'] = self.__create_output_neuron(*args, **kwargs)
        # In order to assign predict output neuron into the same mpi process with the relative output neuron
        self.create_neuron(self.synchronizer.size - len(self.__layer['outputLayer']) % self.synchronizer.size)
        self.__predictLayer['outputLayer'] = self.__create_output_neuron(*args, **kwargs)
        self.__mark_output()
        return self.__layer['outputLayer']

    def create_synapse(self, pre, pos, weight=1.0, conn_spec='one_to_one', model='static_synapse'):
        nest.Connect(pre, pos, conn_spec, syn_spec={'weight': weight, 'model': model})

    def create_random_synapse(self, pre, pos, low_w, high_w, model='stdp_synapse'):
        weights = self.converter.mapping(np.random.rand(len(pre) * len(pos)), 1, 0, high_w, low_w)
        weights = np.array(weights, dtype=np.float).reshape(len(pre), len(pos))
        nest.Connect(pre, pos, conn_spec="all_to_all", syn_spec={'weight': weights.T, 'model': model})

    def link_inputlayer_outputlayer(self):
        self.create_synapse(self.__layer['inputLayer'], self.__layer['outputLayer'], conn_spec="all_to_all", model='stdp_synapse')
        self.create_synapse(self.__predictLayer['inputLayer'], self.__predictLayer['outputLayer'], conn_spec="all_to_all", model='static_synapse')

    def link_inputlayer_outputlayer_random_w(self, low_w, high_w):
        self.create_random_synapse(self.__layer['inputLayer'], self.__layer['outputLayer'], low_w, high_w, model='stdp_synapse')
        self.create_random_synapse(self.__predictLayer['inputLayer'], self.__predictLayer['outputLayer'], low_w, high_w, model='static_synapse')



    def __mark_input(self):
        if 'inputLayer' in self.__layer.keys():
            self.__generator = nest.Create("spike_generator", len(self.__layer['inputLayer']))
            self.__predictGenerator = nest.Create("spike_generator", len(self.__layer['inputLayer']))
            self.__detector['inputDetector'] = nest.Create("spike_detector", params={"to_memory": False})
            self.__multimeter['inputMultimeter'] = nest.Create("multimeter", params={"to_memory": False, 'record_from': ['V_m']})

            '''
            # noise connect inputlayer
            self.__noise_generator = nest.Create("noise_generator", len(self.__layer['inputLayer']))
            self.create_synapse(self.__noise_generator,
                                self.__layer['inputLayer'],
                                model='static_synapse',
                                conn_spec='one_to_one')
            '''

            self.create_synapse(self.__generator,
                                self.__layer['inputLayer'],
                                model='static_synapse',
                                weight=self.__generator_weight)
            self.create_synapse(self.__layer['inputLayer'],
                                self.__detector['inputDetector'],
                                conn_spec='all_to_all',
                                model='static_synapse')
            self.create_synapse(self.__multimeter['inputMultimeter'],
                                self.__layer['inputLayer'],
                                conn_spec='all_to_all',
                                model='static_synapse')

            self.create_synapse(self.__predictGenerator,
                                self.__predictLayer['inputLayer'],
                                model='static_synapse',
                                weight=self.__generator_weight)
            self.create_synapse(self.__predictLayer['inputLayer'],
                                self.__detector['inputDetector'],
                                conn_spec='all_to_all',
                                model='static_synapse')
            self.create_synapse(self.__multimeter['inputMultimeter'],
                                self.__predictLayer['inputLayer'],
                                conn_spec='all_to_all',
                                model='static_synapse')

    def __mark_output(self):
        if 'outputLayer' in self.__layer.keys():
            self.__teacher = nest.Create("spike_generator", len(self.__layer['outputLayer']))
            self.__detector['outputDetector'] = nest.Create("spike_detector", params={"to_memory": True})
            self.__multimeter['outputMultimeter'] = nest.Create("multimeter", params={"to_memory": False, 'record_from': ['V_m']})

            '''
            # noise connect outputlayer
            
            self.__noise_generator = nest.Create("noise_generator", len(self.__layer['outputLayer']))
            self.create_synapse(self.__noise_generator,
                                self.__layer['outputLayer'],
                                model='static_synapse',
                                conn_spec='one_to_one')

            self.create_synapse(self.__noise_generator,
                                self.__predictLayer['outputLayer'],
                                model='static_synapse',
                                conn_spec='one_to_one')

            '''

            self.create_synapse(self.__teacher,
                                self.__layer['outputLayer'],
                                model='static_synapse',
                                weight=1500.)
            self.create_synapse(self.__layer['outputLayer'],
                                self.__detector['outputDetector'],
                                conn_spec='all_to_all',
                                model='static_synapse')
            self.create_synapse(self.__multimeter['outputMultimeter'],
                                self.__layer['outputLayer'],
                                conn_spec='all_to_all',
                                model='static_synapse')

            self.create_synapse(self.__predictLayer['outputLayer'],
                                self.__detector['outputDetector'],
                                conn_spec='all_to_all',
                                model='static_synapse')
            self.create_synapse(self.__multimeter['outputMultimeter'],
                                self.__predictLayer['outputLayer'],
                                conn_spec='all_to_all',
                                model='static_synapse')

    # Setting Input Data
    # --------------------------------------------
    def __set_generator(self, datas, generators):
        for generator, data in zip(generators, datas):
            nest.SetStatus([generator], params={'spike_times': data})

    def __set_input(self, datas):
        self.__set_generator(datas, self.__generator)
        nest.SetStatus(self.__generator, {'origin': nest.GetKernelStatus()['time']})

    def __set_predict_input(self, datas):
        self.__set_generator(datas, self.__predictGenerator)
        nest.SetStatus(self.__predictGenerator, {'origin': nest.GetKernelStatus()['time']})

    def __set_teacher(self, datas):
        self.__set_generator(datas, self.__teacher)
        nest.SetStatus(self.__teacher, {'origin': nest.GetKernelStatus()['time']})

    # Training , Validating and Predicting
    # ---------------------------------------------------
    def __convergence(self, data, target, is_predict=False):
        output = self.__validate(data, is_predict)
        senders = list(output['senders'])
        times = list(output['times'])
        senders = np.array(sum(self.synchronizer.sync(senders), []))
        times = np.array(sum(self.synchronizer.sync(times), []))
        min_index = 0
        if len(senders) != 0:
            min_times = min(times)
            for i, v_time in enumerate(times):
                if v_time == min_times:
                    min_index = i
                    break
        senders -= min(self.__predictLayer['outputLayer'])
        # print(senders, target)
        if len(senders) != 0:
            return len(senders) != 0 and senders[min_index] == target, senders, senders[min_index]
        else:
            return len(senders) != 0 and senders[min_index] == target, senders, -1

    def __validate(self, data, is_predict=False):
        # self.__close_stdp()
        self.__clear_input()
        if not is_predict:
            self.__update_predict_network(self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer']))
        image_spikes = self.converter.data(data)
        self.__clear_detector()
        self.__set_predict_input(image_spikes)
        nest.Simulate(self.__predict_duration)
        # print("Validate:")
        # print(self.__get_output())
        return self.__get_output()

    def train_all(self, datas, targets, iter_max=5, accuracy_rate=.90):
        for i in range(iter_max):
            self.__print_message("Iterating %d", i + 1)
            self.progress_start(len(datas))
            count = 1
            for data, target in zip(datas, targets):
                self.train(data, target)
                self.progress_update(count)
                count += 1
            self.progress_end()
            accuracy = self.__predict_all(datas, targets)
            self.__print_message("Iteration %d : accuracy = %f", i + 1, accuracy)
            if accuracy >= accuracy_rate:
                self.__print_message("Reach accuracy_rate!")
                break

    def train_all_with_result(self, datas, targets, iter_max=5, accuracy_rate=.90):
        accuracy = 0.
        index_l = list()
        for i in range(iter_max):
            self.__print_message("Iterating %d", i + 1)
            self.progress_start(len(datas))
            count = 1
            for data, target in zip(datas, targets):
                self.train(data, target)
                self.progress_update(count)
                count += 1
            self.disconnect()
            self.progress_end()
            accuracy, index_l = self.__predict_all_with_re_index(datas, targets)
            self.__print_message("Iteration %d : accuracy = %f", i + 1, accuracy)
            if accuracy >= accuracy_rate:
                self.__print_message("Reach accuracy_rate!")
                break
        return accuracy, index_l

    def train(self, data, target):
        image_spikes = self.converter.data(data)
        cnt = 0
        while True:
            cnt += 1
            flag_con, result, tt = self.__convergence(data, target)
            if flag_con or cnt > 100:
                break
            teacher_spikes = self.converter.target(target, inh=result)
            self.__train_one(image_spikes, teacher_spikes)
            # print("Train:")
            # print(self.__get_output())

    def test(self, data):
        return self.__convergence(data, 0)[0]

    def __train_one(self, image_spikes, teacher_spikes):
        # self.__open_stdp()
        self.__clear_input()
        self.__clear_detector()
        self.__set_input(image_spikes)
        self.__set_teacher(teacher_spikes)
        nest.Simulate(self.__duration)

    def __predict_all(self, data, target):
        num = 0
        correct = 0
        self.__print_message("Validating:", )
        self.progress_start(len(data))
        for image, label in zip(data, target):
            num += 1
            if self.__convergence(image, label, is_predict=True)[0]:
                correct += 1
            self.progress_update(num)
        self.progress_end()
        return correct / num

    def __predict_all_with_re_index(self, data, target):
        num = 0
        correct = 0
        index = list()
        self.__print_message("Validating:", )
        self.progress_start(len(data))
        for image, label in zip(data, target):
            num += 1
            if self.__convergence(image, label, is_predict=True)[0]:
                correct += 1
                index.append(True)
            else:
                index.append(False)
            self.progress_update(num)
        self.progress_end()
        return correct / num, index

    def predict_all(self, data, target, is_record_error_msg=True, record_file="./record/err_msg.csv"):
        num = 0
        correct = 0

        error_list = []
        predict_error = []

        self.__print_message("Predicting:", )
        self.progress_start(len(data))
        for image, label in zip(data, target):
            convergence_result = self.__convergence(image, label, is_predict=True)
            if convergence_result[0]:
                correct += 1
            else:
                error_list.append(num)
                predict_error.append(convergence_result[2])
            num += 1
            self.progress_update(num)
        self.progress_end()
        self.__print_message("Accuracy in Test Set : %f", correct / num)
        if is_record_error_msg:
            self.__save_predict_error_msg(error_list, predict_error, path=record_file)
        return correct / num

    def predict_all_with_result(self, data, target):
        num = 0
        result_list = []
        self.__print_message("Predicting:", )
        self.progress_start(len(data))
        for image, label in zip(data, target):
            convergence_result = self.__convergence(image, label, is_predict=True)
            result_list.append(convergence_result[2])
            num += 1
            self.progress_update(num)
        self.progress_end()
        return result_list

    def __get_output(self):
        return nest.GetStatus(self.__detector['outputDetector'], 'events')[0]

    def __clear_detector(self):
        nest.SetStatus(self.__detector['outputDetector'], {'n_events': 0})

    def __clear_input(self):
        self.__set_input([np.array([], dtype=np.float)] * len(self.__layer['inputLayer']))
        self.__set_teacher([np.array([], dtype=np.float)] * len(self.__layer['outputLayer']))

    def __update_predict_network(self, conns):
        for conn in conns:
            inv_input = min(self.__predictLayer['inputLayer']) - min(self.__layer['inputLayer'])
            inv_output = min(self.__predictLayer['outputLayer']) - min(self.__layer['outputLayer'])
            nest.SetStatus(nest.GetConnections((conn[0] + inv_input,), (conn[1] + inv_output,)), {'weight': conn[2]})

    def __open_stdp(self):
        pass
        # nest.SetStatus(nest.GetConnections(self.__layer['inputLayer'], self.__layer['outputLayer']), {'tau_plus': 20., 'mu_plus': 1., 'mu_minus': 1.})

    def __close_stdp(self):
        pass
        # nest.SetStatus(nest.GetConnections(self.__layer['inputLayer'], self.__layer['outputLayer']), {'tau_plus': 0., 'mu_plus': 10., 'mu_minus': 10.})

    # Message logging
    def __print_message(self, message, *args):
        if self.synchronizer.rank == 0:
            print(message % args)

    def progress_start(self, n):
        if self.synchronizer.rank == 0:
            self.progressbar.start(n)

    def progress_end(self):
        if self.synchronizer.rank == 0:
            self.progressbar.finish()

    def progress_update(self, n):
        if self.synchronizer.rank == 0:
            self.progressbar.update(n)

    # Network Architecture
    # -------------------------------------------
    def build_network_from_file(self, path):
        conf = pd.read_csv(path, dtype=np.float128)
        synapse = conf[conf['pre'] != 0]
        synapse = synapse[synapse['pos'] != 0]
        synapse = synapse[synapse['pos'] != -1]

        input_neuron = conf[conf['pre'] == 0]['pos']
        output_neuron = conf[conf['pos'] == 0]['pre']

        del conf

        self.create_inputlayer(len(input_neuron))
        self.create_outputlayer(len(output_neuron))

        self.create_synapse(self.__layer['inputLayer'],
                            self.__layer['outputLayer'],
                            conn_spec='all_to_all',
                            model="stdp_synapse")

        self.create_synapse(self.__predictLayer['inputLayer'],
                            self.__predictLayer['outputLayer'],
                            conn_spec='all_to_all',
                            model="static_synapse")

        inv_input = min(self.__predictLayer['inputLayer']) - min(self.__layer['inputLayer'])
        inv_output = min(self.__predictLayer['outputLayer']) - min(self.__layer['outputLayer'])

        conns = synapse.values
        for conn in conns:
            pre = int(conn[1])
            pos = int(conn[0])
            w = float(conn[2])

            nest.SetStatus(nest.GetConnections((pre,), (pos,)), {'weight': w})
            nest.SetStatus(nest.GetConnections((pre + inv_input,), (pos + inv_output,)), {'weight': w})

    def save_network_from_file(self, path):
        n_i_pd = pd.DataFrame({'pre': 0,
                               'pos': np.array(self.__layer['inputLayer'], dtype='int32'),
                               'weight': 1.0
                               })
        n_o_pd = pd.DataFrame({'pre': np.array(self.__layer['outputLayer'], dtype='int32'),
                               'pos': 0,
                               'weight': 1.0
                               })

        n_n_connections = np.array(self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer']),
                                   dtype=np.float128).reshape(-1).tolist()
        n_n_connections = sum(self.synchronizer.sync(n_n_connections), [])
        if self.synchronizer.rank == 0:
            n_n_pd = pd.DataFrame({'pre': np.array(n_n_connections[::3], dtype='int32'),
                                   'pos': np.array(n_n_connections[1::3], dtype='int32'),
                                   'weight': np.array(n_n_connections[2::3], dtype=np.float128)
                                   })
            data_pd = pd.concat([n_i_pd, n_o_pd, n_n_pd])
            data_pd.to_csv(path, index=False)

    def __save_predict_error_msg(self, error_predict_index, error_predict_result, path="./record/err_msg.csv"):
        if self.synchronizer.rank == 0:
            error_msg = pd.DataFrame({'index': np.array(error_predict_index, dtype='int32'),
                                      'result': np.array(error_predict_result, dtype='int32'),
                                      })
            error_msg.to_csv(path, index=False)

    @staticmethod
    def __get_connections(source, target):
        return nest.GetStatus(nest.GetConnections(source, target), ['source', 'target', 'weight'])

    def disconnect(self):
        change = self.__disconnect_rate
        cc = self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer'])
        cc_input_dis = min(self.__predictLayer['inputLayer']) - min(self.__layer['inputLayer'])
        cc_output_dis = min(self.__predictLayer['outputLayer']) - min(self.__layer['outputLayer'])
        for c in cc:
            if abs(c[2] - 1.0) <= change:
                nest.Disconnect(pre=[c[0]], post=[c[1]], conn_spec={'rule': "one_to_one"}, syn_spec={'model': 'stdp_synapse'})
                nest.Disconnect(pre=[c[0] + cc_input_dis], post=[c[1] + cc_output_dis], conn_spec={'rule': "one_to_one"}, syn_spec={'model': 'static_synapse'})

    def get_rest_connections_num(self):
        cc = self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer'])
        num = [len(cc)]
        num = sum(self.synchronizer.sync(num), [])
        return sum(num)

    # Visualization
    # -------------------------------------------
    def show(self):
        pylab.show()

    def get_input_spikes(self):
        nest.raster_plot.from_device(self.__detector['inputDetector'], hist=False)

    def get_output_spikes(self):
        nest.raster_plot.from_device(self.__detector['outputDetector'], hist=False)

    def get_output_trace(self):
        nest.voltage_trace.from_device(self.__multimeter['outputMultimeter'])

    def get_input_trace(self):
        nest.voltage_trace.from_device(self.__multimeter['inputMultimeter'])
