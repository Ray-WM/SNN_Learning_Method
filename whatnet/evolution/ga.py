"""
遗传算法
"""

from whatnet.evolution.nests import Nests
import random
import numpy as np
import copy


class GA(object):
    def __init__(self, snn_net, arr_kernel, init_fitness, popu_num=50, generation_num=20, cross_rat=0.7, mutation_rat=0.05):
        """
        初始化
            :param popu_num: 种群大小
            :param generation_num: 迭代次数
            :param cross_rat: 交叉概率
            :param mutation_rat: 变异概率
            :param popu_list: 种群列表
        """
        self.popu_num = popu_num
        self.generation_num = generation_num
        self.cross_rat = cross_rat
        self.mutation_rat = mutation_rat
        self.max_list = []
        self.init_nest = Nests(arr_kernel=arr_kernel)
        self.max_item = {"DNA": self.init_nest.DNA, "fitness": init_fitness, "item": self.init_nest}
        # 初始化种群
        self.popu_list = []
        self.fit_list = []
        self.cross_point = 0
        self.num = 0
        self.max_num = 0

        self.snn_net = snn_net
        # for _ in range(self.popu_num):
        #     self.popu_list.append(nests(arr_kernel=arr_kernel))
        """
                      [[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]
                
                """
        kernel_list = [arr_kernel]
        index_item = 0
        for i in range(len(arr_kernel)):
            for j in range(len(arr_kernel[i])):
                if arr_kernel[i][j][0] == 1 and j != 0 and j != 7:
                    index_item = i
                    break
        # 获取卷积和中不为0的一条数据
        for item in range(len(arr_kernel[index_item])):
            if arr_kernel[index_item][item][0] == 1:
                copy_kernel = copy.deepcopy(arr_kernel)
                copy_kernel[0][item - 1] = copy_kernel[0][item]
                copy_kernel[0][item + 1] = copy_kernel[0][item]
                """
                      [[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]
                
                """
                kernel_list.append(copy_kernel)
                copy_kernel = copy.deepcopy(arr_kernel)
                copy_kernel[0][item - 2] = copy_kernel[0][item]
                copy_kernel[0][item + 2] = copy_kernel[0][item]
                """
                      [[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0]]
                
                """
                kernel_list.append(copy_kernel)
                copy_kernel = copy.deepcopy(arr_kernel)
                copy_kernel[0][item - 1] = copy_kernel[0][item]
                copy_kernel[0][item + 1] = copy_kernel[0][item]
                if item == 4:
                    copy_kernel[0][0] = copy_kernel[0][item]
                else:
                    copy_kernel[0][7] = copy_kernel[0][item]
                """
                    [[1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0]]
              
              """
                kernel_list.append(copy_kernel)
                copy_kernel = copy.deepcopy(arr_kernel)
                copy_kernel[0][0] = copy_kernel[0][item]
                copy_kernel[0][1] = copy_kernel[0][item]
                if item == 4:
                    copy_kernel[0][item - 1] = copy_kernel[0][item]
                else:
                    copy_kernel[0][item + 1] = copy_kernel[0][item]
                copy_kernel[0][6] = copy_kernel[0][item]
                copy_kernel[0][7] = copy_kernel[0][item]
                """
                    [[1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1]]
              
              """
                kernel_list.append(copy_kernel)
                break
        for _ in range(self.popu_num):
            self.popu_list.append(Nests(random.choice(kernel_list)))

    def get_fitness(self):
        self.fit_list = []
        for item in self.popu_list:
            self.fit_list.append(item.get_fitness(self.snn_net))

    def select(self):
        """
        筛选操作：轮盘赌法
        :return:
        """
        total_fitness = sum(self.fit_list)
        new_fit_list = [sum(self.fit_list[:i + 1]) / total_fitness for i in range(len(self.fit_list))]

        select_rat_list = [random.random() for _ in range(len(self.popu_list))]
        select_rat_list.sort()
        new_popu_list = self.popu_list

        fit_index = 0
        new_index = 0
        # 挑选最优个体进入下一代种群中
        max_fit = max(self.fit_list)
        if self.generation_num / 4 * (self.cross_point + 1) < self.num:
            self.cross_point += 1
            for index in range(len(self.fit_list)):
                new_popu_list[new_index] = self.max_item['item']
        else:
            for index in range(len(self.fit_list)):
                if self.fit_list[index] == max_fit:
                    if max_fit > self.max_item["fitness"]:
                        self.max_item['DNA'] = self.popu_list[index].DNA
                        self.max_item['fitness'] = max_fit
                        self.max_item['item'] = self.popu_list[index]
                    new_popu_list[new_index] = self.popu_list[index]
                    self.max_list.append({"DNA": self.popu_list[index].DNA, "fitness": max_fit})
                    new_index += 1
            self.max_num = new_index
        while new_index < len(self.popu_list):
            if select_rat_list[new_index] < new_fit_list[fit_index]:
                new_popu_list[new_index] = self.popu_list[fit_index]
                new_index = new_index + 1
            else:
                fit_index = fit_index + 1

        self.popu_list = new_popu_list

    def crossover(self):
        """
        个体之间进行交叉
        :return:
        """
        if self.max_num < 5:
            start_point = self.max_num
        else:
            start_point = 5
        for i in range(start_point, len(self.popu_list) - 1):
            if random.random() < self.cross_rat:
                # 选择交叉点(暂定选择四个交叉点)
                skip_num = 8 * 8
                point_list = [7 + i * 8 + self.cross_point * skip_num for i in range(8)]
                # cross_point_num=random.randint(skip_num*self.cross_point,skip_num*(self.cross_point+1))
                # 获取四个交叉点
                cross_point_num = random.choice(point_list)
                # 根据交叉点进行交叉操作
                DNA_one = self.popu_list[i].DNA[:cross_point_num] + self.popu_list[i + 1].DNA[cross_point_num:]
                DNA_two = self.popu_list[i + 1].DNA[:cross_point_num] + self.popu_list[i].DNA[cross_point_num:]
                self.popu_list[i].DNA = DNA_one
                self.popu_list[i + 1].DNA = DNA_two

    def mutation(self):
        """
        变异操作
        :return:
        """
        if self.max_num < 5:
            start_point = self.max_num
        else:
            start_point = 5
        for i in range(start_point, len(self.popu_list)):
            for index in range(self.cross_point * 8 * 8, (self.cross_point + 1) * 8 * 8):
                if self.is_mutation_point(index, i):
                    if random.random() < self.mutation_rat:
                        if self.popu_list[i].DNA[index] == '1':
                            self.popu_list[i].DNA = self.popu_list[i].DNA[:index] + '0' + self.popu_list[i].DNA[index + 1:]
                        else:
                            self.popu_list[i].DNA = self.popu_list[i].DNA[:index] + '1' + self.popu_list[i].DNA[index + 1:]

    def is_mutation_point(self, index, item):
        dnaList = [int(i) for i in self.popu_list[item].DNA]
        kernels = np.array(dnaList).reshape(4, 8, 8)
        kernel = kernels[self.cross_point]
        index_x = int((index - self.cross_point * 8 * 8) / 8)
        index_y = (index - self.cross_point * 8 * 8) % 8

        if index_x != 0 and index_y != 0 and index_x != 7 and index_y != 7:
            if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != \
                    kernel[index_x][index_y] or kernel[index_x - 1][index_y] != kernel[index_x][index_y]:
                return True
            else:
                return False
        if index_x == 0:
            if index_y != 0 and index_y != 7:
                if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != \
                        kernel[index_x][index_y]:
                    return True
                else:
                    return False
            if index_y == 0:
                if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != kernel[index_x][index_y]:
                    return True
                else:
                    return False
            if index_y == 7:
                if kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != kernel[index_x][index_y]:
                    return True
                else:
                    return False
        if index_x == 7:
            if index_y != 0 and index_y != 7:
                if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x - 1][index_y] != \
                        kernel[index_x][index_y]:
                    return True
                else:
                    return False
            if index_y == 0:
                if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x - 1][index_y] != kernel[index_x][index_y]:
                    return True
                else:
                    return False
            if index_y == 7:
                if kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x - 1][index_y] != kernel[index_x][index_y]:
                    return True
                else:
                    return False
        if index_y == 0:
            if kernel[index_x][index_y + 1] != kernel[index_x][index_y] or kernel[index_x - 1][index_y] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != \
                    kernel[index_x][index_y]:
                return True
            else:
                return False
        if index_y == 7:
            if kernel[index_x][index_y - 1] != kernel[index_x][index_y] or kernel[index_x - 1][index_y] != kernel[index_x][index_y] or kernel[index_x + 1][index_y] != \
                    kernel[index_x][index_y]:
                return True
            else:
                return False

    def run(self):
        for index in range(self.generation_num):
            self.get_fitness()
            self.select()
            if self.cross_point >= 4:
                break
            self.crossover()
            self.mutation()
            self.num += 1
            if self.snn_net.synchronizer.rank == 0:
                print(self.max_item["DNA"], self.max_item["fitness"])


if __name__ == '__main__':
    arr_kernal = np.array([[[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]],
                           [[0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0]],
                           [[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]]])
    # 获取模型的4个卷积核 array.shape(4,8,8)
    init_fit = 0.7  # 训练后模型的准确率
    GA_item = GA(arr_kernal, init_fit, popu_num=50, generation_num=100, cross_rat=0.7, mutation_rat=0.05)  # 在迭代的过程中会判断是否提前结束
    GA_item.run()
