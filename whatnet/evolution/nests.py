"""nestsl：代表迭代过程中种群的个体

   属性：
                   DNA:(str)个体的编码之后的信息
            infomation:转码之后dna所蕴含信息的表现型

   方法：
            encode：将dna转换成矩阵
            decode：将矩阵转换成dna
        getfitness：获取个体的适应度

"""
import numpy as np
import whatnet.strategy.offspring_fitness as ofitness


class Nests(object):
    def __init__(self, arr_kernel):
        """
        个体初始化：
            随机生成字符串或者是初始过程中获取
        """
        self.encode(arr_kernel)
        self.fitness = 0
        self.arr_kernel = arr_kernel

    def decode(self):
        """
        将DNA转换成4个8*8的矩阵并存处在list 中
        :return: np.array()
        """
        dna_list = [int(i) for i in self.DNA]
        self.arr_kernel = np.array(dna_list).reshape(4, 8, 8)
        # return self.information

    def encode(self, arr_kernel):
        """将矩阵转换为字符串"""
        self.DNA = "".join(str(i) for i in arr_kernel.reshape(1, 256)[0].tolist())

    def set_DNA(self, arr_kernel):
        self.encode(arr_kernel)

    def get_fitness(self, net):
        self.decode()
        self.fitness = self.nest(self.arr_kernel, net)
        return self.fitness

    def nest(self, info, net):  # 求适应度的函数
        if net.synchronizer.rank == 0:
            print("DNA:", self.DNA)
        return ofitness.get_fitness(net, info)
