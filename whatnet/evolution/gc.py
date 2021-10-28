import whatnet.strategy.offspring_fitness as oft
from whatnet.common.mpi_sync import Synchronizer
import numpy as np
import json
import random


class GC(object):
    def __init__(self, kernel_size):
        self.pop = []
        self.current_pop = []
        self.best_pop = []
        self.kernel_size = kernel_size
        self.current = 1
        self.init_pop(kernel_size)
        self.syn = Synchronizer()

    def init_pop(self, kernel_size):
        for _ in range(kernel_size ** 2):
            dna = [0] * (kernel_size ** 2)
            dna[_] = 1
            self.pop.append({"DNA": "".join(str(x) for x in dna), "Fitness": random.random(), "Index": _})
        self.current_pop = self.pop

    def get_fitness(self, dna):
        kernel = np.array([int(x) for x in dna]).reshape(self.kernel_size, self.kernel_size)
        return oft.get_fitness_gc(dna, kernel, train_set_num=500, test_set_num=1000)
        # return random.random()

    def evaluate_pop(self):
        for i in range(len(self.pop)):
            self.pop[i]["Fitness"] = self.get_fitness(self.pop[i]["DNA"])

    def next_generation(self, is_first_generation):
        self.pop.sort(key=lambda x: x["Fitness"])
        if not is_first_generation:
            self.best_pop.append(self.pop[-2:])
        self.current_pop = []
        dna_set = set()
        for item in self.pop[-2:]:
            for _ in range(self.kernel_size ** 2):
                if item["DNA"][_] != "1":
                    dna = [int(x) for x in item["DNA"]]
                    dna[_] = 1
                    dna = "".join(str(x) for x in dna)
                    if dna not in dna_set:
                        dna_set.add(dna)
                        self.current_pop.append({"DNA": dna, "Fitness": 0.})
        self.current += 1
        self.pop = self.current_pop

    def gc(self, is_record=False, path="./record/gc_result.json"):
        self.current = 1
        for i in range(5):
            if i == 0:
                self.next_generation(True)
            else:
                self.evaluate_pop()
                self.next_generation(False)
        if self.syn.rank == 0:
            record = {}
            name = lambda x: "Generation " + str(x)
            for index, item in enumerate(self.best_pop):
                print("Generation : ", index + 1)
                print(item)
                record[name(index)] = item
            if is_record:
                with open(path, 'w') as f:
                    json.dump(record, f, indent=2)

    def get_fitness_x(self, dna):
        kernel = np.array([float(x) / 10 for x in dna]).reshape(self.kernel_size, self.kernel_size)
        return oft.get_fitness_gc(dna, kernel, train_set_num=500, test_set_num=1000)

    def evaluate_pop_x(self):
        for i in range(len(self.pop)):
            self.pop[i]["Fitness"] = self.get_fitness_x(self.pop[i]["DNA"])

    def next_generation_x(self, best):
        self.pop.sort(key=lambda x: x["Fitness"])
        self.current_pop = []
        for _ in range(self.kernel_size ** 2):
            if best["DNA"][_] == "0":
                dna = [int(x) for x in best["DNA"]]
                dna[_] = 1
                dna = "".join(str(x) for x in dna)
                self.current_pop.append({"DNA": dna, "Fitness": 0., "Index": _})
        self.pop = self.current_pop

    def gc_x(self, is_record=False, path="./record/gc_result.json"):
        self.current = 1
        record = {}
        for _ in range(5):
            self.evaluate_pop_x()
            self.pop.sort(key=lambda x: x["Fitness"])
            best = self.pop[-1]
            while True:
                best_dna = [int(x) for x in best["DNA"]]
                best_dna[best["Index"]] += 1
                if best_dna[best["Index"]] > 10:
                    break
                best_dna = "".join(str(x) for x in best_dna)
                fitness = self.get_fitness_x(best_dna)
                if fitness < best["Fitness"]:
                    self.next_generation_x(best)
                    self.current += 1
                    if self.syn.rank == 0:
                        record[self.current] = best
                        print("Generation : ", self.current)
                        print(best)
                    break
                else:
                    best["DNA"] = best_dna
                    best["Fitness"] = fitness
        if is_record and self.syn.rank == 0:
            with open(path, 'w') as f:
                json.dump(record, f, indent=2)


if __name__ == "__main__":
    gc = GC(4)
    gc.gc(is_record=True)
