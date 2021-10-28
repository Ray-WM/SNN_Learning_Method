import matplotlib.pyplot as plt
import nest
import logging

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


class NetworkBase(object):
    """A Network only contains neurons(iaf_psc_alpha).
    
    There also are other devices in a network:
        generator(spike_generator): puts input data to input neurons
        detector(spike_detector): records spike of output neurons
        multimeter(multimeter): records the the membrane voltages of output neurons
    """
    
    def __init__(self, converter):
        self.__nest_init()
        
        self.__generator = None
        self.__teacher = None
        self.__detector = nest.Create('spike_detector', params={'withgid': True, 'withtime': True, 'precise_times': True})
        self.__multimeter = nest.Create('wn_multimeter', params={'withtime': True, 'record_from': ['V_m']})
        self.__output_layer = None
        
        self._figure = 1
        self.__duration = 350.
        self._converter = converter
        
    def __nest_init(self):
        nest.ResetKernel()
        nest.CopyModel("iaf_psc_alpha", "wn_neuron")
        nest.CopyModel('stdp_synapse', 'wn_synapse')
        
        nest.CopyModel('spike_generator', 'wn_generator')
        #nest.CopyModel('poisson_generator', 'wn_generator')
        nest.CopyModel('spike_detector', 'wn_detector')
        nest.CopyModel('multimeter', 'wn_multimeter')
    
    def create_neuron(self, *args, **kw):
        return nest.Create('wn_neuron', *args, **kw)
    
    def create_synapse(self, pre, pos, 
                       weight=1.0, 
                       conn_spec='one_to_one', 
                       model='wn_synapse'):
        
        nest.Connect(pre, pos, conn_spec, 
                     syn_spec={'weight': weight, 'model': model})
        
    def mark_input(self, input_neurons):
        self.__generator = nest.Create('wn_generator', len(input_neurons))
        self.create_synapse(self.__generator, 
                            input_neurons, 
                            model='static_synapse')
    
    def mark_output(self, output_neurons):
        
        self.create_synapse(output_neurons, 
                            self.__detector, 
                            conn_spec='all_to_all',
                            model='static_synapse')
        self.create_synapse(self.__multimeter, 
                            output_neurons,
                            conn_spec='all_to_all',
                            model='static_synapse')
        self.__teacher = nest.Create('wn_generator', len(output_neurons))
        self.create_synapse(self.__teacher, output_neurons, model='static_synapse')
        nest.Connect(output_neurons, output_neurons, conn_spec={'rule': 'all_to_all', 'autapses': False}, syn_spec={'weight': -10.})
        self.__output_layer = output_neurons
        
    def __set_generator(self, datas, generators):
        for generator, data in zip(generators, datas):
            nest.SetStatus([generator], params={'spike_times': data, 'spike_weights': [2000.] * len(data)})
            
    def __set_input(self, datas):
        datas = self._converter.data(datas)
        self.__set_generator(datas, self.__generator)
        
    def __set_teacher(self, datas):
        self.__set_generator(datas, self.__teacher)
    
    def get_output(self):
        return nest.GetStatus(self.__detector, 'events'), nest.GetStatus(self.__multimeter, 'events')
    
    def get_connections(self):
        return nest.GetStatus(nest.GetConnections(synapse_model="wn_synapse"),["source", "target", "weight"]),  \
               nest.GetStatus(nest.GetConnections(self.__generator), ["source", "target", "weight"]), \
               nest.GetStatus(nest.GetConnections(self.__multimeter), ["source", "target", "weight"])
            
    def show(self):
        # Plot results
        # This is a temp function 
        # Visualization will be implemented in other module
        sp, me = self.get_output()
        sp, me = sp[0], me[0]
        evs = sp["senders"]
        ts = sp["times"]
        plt.figure(self._figure)
        self._figure += 1
        plt.ylim(100, 112)
        plt.plot(ts, evs, '.')
        
        
        plt.figure(self._figure)
        self._figure += 1

        for i in range(10):
            Vms1 = me["V_m"][i::10]
            ts1 = me["times"][i::10]
            plt.plot(ts1, Vms1, label="Target : " + str(i))
        plt.legend()

    def __convergence(self, data, target):
        self.predict(data)
        ts = self.get_output()[0][0]
        #return len(ts['senders']) and (ts['senders'][0] == self.__output_layer[target])
        return self.__output_layer[target] in ts['senders']

    def train(self, data, target):
        self.__set_input(data)
        cnt = 0
        while not self.__convergence(data, target):
            cnt += 1
            logging.info("Training: #%s", cnt)
            self.__set_teacher(self._converter.target(target))
            nest.SetStatus(self.__generator, {'origin': nest.GetKernelStatus()['time']})
            nest.SetStatus(self.__teacher, {'origin': nest.GetKernelStatus()['time']})
            nest.Simulate(self.__duration)
        
    def predict(self, data):
        self.__set_input(data)
        self.__set_teacher([])
        nest.SetStatus(self.__detector, {'record_to': ['memory'], 'n_events': 0})
        nest.SetStatus(self.__multimeter, {'record_to': ["memory"], 'n_events': 0})
        nest.SetStatus(self.__generator, {'origin': nest.GetKernelStatus()['time']})
        nest.Simulate(self.__duration)
        