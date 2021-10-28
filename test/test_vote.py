import numpy as np
import random


def vote(result, target_list):
    r = np.array(result)
    r = r.T
    vote_result = []

    # Get vote result
    for _ in r:
        b = _.tolist()
        vote_result.append(max(set(b), key=b.count))

    count = 0

    for vote_r, target in zip(vote_result, target_list):
        if vote_r == target:
            count += 1

    return count / len(target_list)

def select(p_list):
    ran = random.random()
    i_list = np.arange(0, len(p_list))
    sum_p = 0
    for index, p in zip(i_list, p_list):
        sum_p += p
        if ran < sum_p:
            return index


def create_training_set(p_list):
    p_train_index = list()
    for _ in range(1000):
        i_p = select(p_list)
        p_train_index.append(i_p)
    return p_train_index

if __name__ == "__main__":

    a = [[-1, -1], [-1, -1], [-1, -1]]
    target_ss = [-1, -1, -1]

    p_l = [1/1000]*1000

    pp = create_training_set(p_l)
    print(pp)
    s_p = set(pp)
    print(len(s_p))
