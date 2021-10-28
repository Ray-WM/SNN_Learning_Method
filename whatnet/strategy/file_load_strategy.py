from whatnet.networkbase import NetworkBase
import pandas as pd
import numpy as np

"""可用于参考Network的使用方式
"""


class FileLoadStrategy:
    """Build a network from file config(csv)
    
    Format of file config:
        0,input_1,weight
        0,input_2,weight
        ...
        0,input_n,weight
        
        output_1,0,weight
        output_2,0,weight
        ...
        output_m,0,weight
        
        pre_1,pos_1,weight_1
        pre_2,pos_2,weight_2
        ...
        pre_k,pos_k,weight_k
    """

    def build(self, path):
        net = NetworkBase()

        conf = pd.read_csv(path)
        synapse = conf[conf['pre'] != 0]
        synapse = synapse[synapse['pos'] != 0]

        ids = set(conf.values.flatten()) - {0}
        neurons = net.create_neuron(len(ids))
        id2neurons = dict(zip(ids, neurons))

        net.create_synapse([id2neurons[pre] for pre in synapse['pre'].values],
                           [id2neurons[pos] for pos in synapse['pos'].values],
                           weight=synapse['weight'].values)

        # connect devices with neurons
        input_neurons = conf[conf['pre'] == 0]['pos'].values
        net.mark_input([id2neurons[_] for _ in input_neurons])
        output_neurons = conf[conf['pos'] == 0]['pre'].values
        # net.mark_output([id2neurons[_] for _ in output_neurons])
        net.mark_output(neurons)

        return net

    def save(self, network, path):
        pass

    def save_network_architecture(self, network, path):
        conn_neuron_to_neuron, conn_input_neuron, conn_output_neuron = network.get_connections()

        bias = 2

        # 获取神经元之间的链接
        n_n_pd = pd.DataFrame(np.array(conn_neuron_to_neuron), columns=["pre", "pos", "weight"])
        n_n_pd["pre"], n_n_pd["pos"] = n_n_pd["pre"].astype(int) - bias, n_n_pd["pos"].astype(int) - bias

        # 获取输入神经元与generator的链接
        n_i_pd = pd.DataFrame(np.array(conn_input_neuron), columns=["pre", "pos", "weight"])
        n_i_pd["pre"], n_i_pd["pos"] = 0, n_i_pd["pos"].astype(int) - bias

        # 获取输出神经元与multimeter的链接
        n_o_pd = pd.DataFrame(np.array(conn_output_neuron), columns=["pre", "pos", "weight"])
        n_o_pd["pre"], n_o_pd["pos"] = n_o_pd["pos"].astype(int) - bias, 0

        # 合并
        data_pd = pd.concat([n_i_pd, n_o_pd, n_n_pd])
        data_pd.to_csv(path, index=False)
