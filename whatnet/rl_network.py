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
class RlNetwork(NetworkBase):
    def __init__(self, converter: CnnConverter):
        
        self.__nest_init()
        self.__layer = {}
        self.__noise_generator = {}
        self.__generator = {}

        # 脉冲信息获取器，detector和万用表，万用表一般不适用
        self.__detector = {}
        self.__multimeter = {}

        self.__duration = 500.
        
        self.__generator_weight = 200.
        self.converter = converter

        self.synchronizer = Synchronizer()

        self.progressbar = progressbar.ProgressBar()

        self.__disconnect_rate = 1.0

        #强化学习参数
        
        # self.n_state = 36     #状态个数  
        self.alpha = 0.01       #网络参数的学习率
        self.action = ['dec','unc', 'inc']      #动作的选取(decrease减少，unchanged不变，increase增加)
        self.n_action = len(self.action)        #动作空间的大小
        self.alpha_ql = 0.1     #q-learning的学习率
        self.gamma = 0.9        #q-learning的参数gamma
        self.max_episodes = 10000000     #训练次数
        self.finish_flag = "OVER"       #结束标记
        self.__env_init()



    def set_disconnect_rate(self, rate):
        self.__disconnect_rate = rate

    # 自定义的初始化网络
    def __nest_init(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.CopyModel("iaf_psc_alpha", "input_neuron", params={"V_th": -68., "t_ref": 20.})
        nest.CopyModel("iaf_psc_alpha", "output_neuron", params={"V_th": -67., "t_ref": 300.})
        nest.CopyModel("iaf_psc_alpha", "wn_neuron")

    #创建网络
    def create_neuron(self, *args, **kwargs):
        return nest.create_neuron("wn_neuron", *args, *kwargs)

    @staticmethod
    def __create_output_neuron(*args, **kwargs):
        return nest.Create("output_neuron", *args, **kwargs)

    @staticmethod
    def __create_input_neuron(*args, **kwargs):
        return nest.Create("input_neuron", *args, **kwargs)


    def create_inputlayer(self, *args, **kwargs):
        self.__layer['inputlayer'] = self.__create_input_neuron(*args, *kwargs)
        self.__mark_input()
        return self.__layer['inputLayer']

    def create_outputlayer(self, *args, **kwargs):
        self.__layer['outputLayer'] = self.__create_output_neuron(*args, *kwargs)
        self.create_neuron(self.synchronizer.size - len(self.__layer['outputLayer']) % self.synchronizer.size)
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
        

    def link_inputlayer_outputlayer_random_w(self, low_w, high_w):
        self.create_random_synapse(self.__layer['inputLayer'], self.__layer['outputLayer'], low_w, high_w, model='stdp_synapse')
        

    #记录输入层神经元的信息
    def __mark_input(self):
         if 'inputLayer' in self.__layer.keys():
            self.__generator = nest.Create("spike_generator", len(self.__layer['inputLayer']))
            self.__detector['inputDetector'] = nest.Create("spike_detector", params={"to_memory":False})
            self.__multimeter['inputMultimeter'] = nest.Create("multimeter", params={"to_memory": False, 'record_from': ['V_m']})
            #创建突触
            #脉冲发射层和input层的突触连接
            self.create_synapse(self.__generator,
                                self.__layer['intputLayer'],
                                model = 'static_synapse',
                                weight = self.__generator_weight)

            #建立input层和探测层的突触连接
            self.create_synapse(self.__layer['inputLayer'],
                                self.__detector,
                                conn_spec = 'all_to_all',
                                model = 'static_synapse')

            #建立万用表层和input层的突触连接
            self.create_synapse(self.__multimeter['inputMultimeter'],
                                self.__layer['inputLayer'],
                                conn_spec = 'all_to_all',
                                model = 'static_synapse')
    
    def __mark_output(self):
        if 'outputLayer' in self.__layer.keys():
            #定义了记录输出层神经元的detector和万用表
            self.__detector['outputLayer'] = nest.Create("spike_detector", params={"to_memory": True})
            self.__multimeter['outputLayer'] = nest.Create("multimeter", params={"to_memory":False, 'record_from': ['V_m']})

            #建立output层和det层的突触连接
            self.create_synapse(self.__layer['outputLayer'],
                                self.__detector,
                                conn_spec = 'all_to_all',
                                model = 'static_synapse')

            #建立万用表和op层的连接
            self.create_synapse(self.__multimeter,
                                self.__layer['outputLayer'],
                                conn_spec = 'all_to_all',
                                model = 'static_synapse')

# -------------------设置输入信号-------------------
    #设置脉冲发射器
    def __set_generator(self, datas, generators):
        for generator, data in zip(generators, datas):
            nest.SetStatus([generator], params = {'spike_time': data})
    #设置input层
    def __set_input(self, datas):
        self.__set_generator(datas, self.__generator)
        nest.SetKernelStatus(self.generator, {'origin': nest.GetKernelStatus()['time']})

# ------------------RL训练------------------

    # # 初始化神经网络
    # def __env_init(self, path):
    #     build_network_from_file(path)
    #     print('OK')

    # # 突触选择
    # def __synapse_get(self, path, i, j):
    #     conf = pd.read_csv(path, dtype=np.float128)
        
    # # 权值变化
    # def __synapse_change(self, weight, base_action):
    #     weight = weight + sum(base_action) * weight
    #     return weight


    # # 突触实施action
    # def __net_step(self, action):
    #     # 基准
    #     base_action = np.array([0, 0, 0])
    #     if action == 0:     #减少
    #         base_action[0] = -alpha
    #     if action == 1:     #不变
    #         base_action[1] = 0
    #     if action == 2:     #增加
    #         base_action[2] = alpha

    def __get_output(self):
        return nest.GetStatus(self.__detector['outputDetector'], 'e')

# -----------------O-V-E-R------------------

# 网络构造
#-------------------------------------------------------
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
        # 构建静态突触连接input层和output层
        self.create_synapse(self.__layer['inputLayer'],
                            self.__layer['outputLayer'],
                            conn_spec='all_to_all',
                            model='static_synapse')

        conns = synapse.values
        for conn in conns:
            pre = int(conn[1])
            pos = int(conn[0])
            w = float(conn[2])

            nest.SetStatus(nest.GetConnections((pre,), (pos,)), {'weight' : w})

    def save_network_from_file(self, path):
        n_i_pd = pd.DataFrame({ 'pre': 0,
                                'pos': np.array(self.__layer['inputLayer'], dtype='int32'),
                                'weight': 1.0
                                })
        n_o_pd = pd.DataFrame({ 'pre': np.array(self.__layer['outputLayer'], dtype='int32'),
                                'pos': 0,
                                'weight': 1.0
                                })

        n_n_connections = np.array( self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer']),
                                    dtype=np.float128).reshape(-1).tolist()
        n_n_connections = sum(self.synchronizer.sync(n_n_connections), [])
        if self.synchronizer.rank == 0:
            n_n_pd = pd.DataFrame( {'pre': np.array(n_n_connections[::3], dtype='int32'),
                                    'pos': np.array(n_n_connections[1::3], dtype='int32'),
                                    'weight': np.array(n_n_connections[2::3], dtype=np.float128)
                                    })
            data_pd = pd.concat([n_i_pd, n_o_pd, n_n_pd])
            data_pd.to_csv(path, index=False)

    @staticmethod
    # 建立链接
    def __get_connections(source, target):
        return nest.GetStatus(nest.GetConnections(source, target), ['source', 'target', 'weight'])

    # 断开链接
    def disconnect(self):
        change = self.__disconnect_rate
        cc = self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer'])
        cc_input_dis = min(self.__predictLayer['inputLayer']) - min(self.__layer['inputLayer'])
        cc_output_dis = min(self.__predictLayer['outputLayer']) - min(self.__layer['outputLayer'])
        for c in cc:
            if abs(c[2] - 1.0) <= change:
                nest.Disconnect(pre=[c[0]], post=[c[1]], conn_spec={'rule': "one_to_one"}, syn_spec={'model': 'stdp_synapse'})
                nest.Disconnect(pre=[c[0] + cc_input_dis], post=[c[1] + cc_output_dis], conn_spec={'rule': "one_to_one"}, syn_spec={'model': 'static_synapse'})

    # 统计休眠连接
    def get_rest_connections_num(self):
        cc = self.__get_connections(self.__layer['inputLayer'], self.__layer['outputLayer'])
        num = [len(cc)]
        num = sum(self.synchronizer.sync(num), [])
        return sum(num)


