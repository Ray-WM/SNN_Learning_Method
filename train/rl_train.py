from whatnet import rl_network
import numpy as np




class Env(object):
    def __init__(self, path, cnn):
        self.action_sapce = ['dec', 'equ', 'inc']
        self.n_action = len(self.action_sapce)
        self.weight_path = path
        self.__weight = np.loadtxt(open(self.path, "rb"),delimiter=",",skiprows=0)
        self.base_state = [1,0,1,0]
        self.__net = rl_network(cnn)
        self.__alpha = 0.01
        self.__data = []
        self.__target = []
        self.build_net()
        self.__first_senders = self.build_net(self.weight_path)[0]
        self.__first_time = self.build_net(self.weight_path)[1]
        # 记录激活的神经元和其激活的时间
        self.__mark_all = {str(np.argmin(self.__first_time)+1): np.min(self.__first_time)}
        # 记录所有最先激发的神经元的激发时间
        self.__mark_time = [np.min(self.__first_time)]
        # 记录目标神经元的激活时间
        self.__mark_diff_time = [self.__first_time[self.__target]]
        

    def build_net(self,path):
        # build net
        self.__net.create_inputlayer(28 * 28)
        self.__net.create_outputlayer(10)
        self.__net.link_inputlayer_outputlayer()
        self.__net.build_network_from_file(path)
        self.__net.train(self.__data, self.__target)
        return [self.__net.__get_opobj()[0], self.__net.__get_opobj()[1]]

    

    def step(self, action, i, j, s=[1,0,1,0]):
        s_new = s
        flag = 1
        # self.__syp_weight = self.__weight[i][j]
        if action == 0:     # 减小权值,得到输出
            self.__weight[i][j] -= -self.__alpha * self.__weight[i][j]
            np.savetxt(self.weight_path, self.__weight, delimiter=',')
            
        elif action == 1:   # 权值不变，得到输出
            self.__weight[i][j] -= 0 * self.__weight[i][j]
            np.savetxt(self.weight_path, self.__weight, delimiter=',')
       
        elif action == 2:   # 增加权值，得到输出
            self.__weight[i][j] -= self.__alpha * self.__weight[i][j]
            np.savetxt(self.weight_path, self.__weight, delimiter=',')


        # 变更环境
        self.__net.build_network_from_file(self.weight_path)
        self.__net.train(self.__data, self.__target)
        #sender = self.__net.__get_opobj()[0]
        
        # 记录修改一次参数后的神经元激发参数
        time = self.__net.__get_opobj()[1]
        first_sender = np.argmin(time)
        self.__mark_all[str(first_sender)] = np.min(time)
        # self.__mark_diff_time.append(time[self.__target])
        
        # 奖励函数
        if first_sender == self.__target:
            if np.min(time) - self.__mark_time[-1] < 0:
                reward = 1
                self.__mark_time.append(np.min(time))
            
            elif np.min(time) - self.__mark_time[-1] >= 0:
                reward = -1 
                self.__mark_time.append(np.min(time))
        
        else:
            if time[self.__target] - np.min(time) < self.__mark_diff_time[-1] - self.__mark_time[-1]:
                reward = 1
                self.__mark_diff_time.append(time[self.__target])
            else:
                reward = -1
                self.__mark_diff_time.append(time[self.__target])

        # 更新状态
        s_new = [s_new[2], s_new[3], action, reward]

        return s_new, reward



