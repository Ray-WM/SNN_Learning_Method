import pandas as pd
import numpy as np
import collections

def alpha(err_p):
    return 0.5 * np.log((1 - err_p) / err_p)

def disconnect(self, id):
    change = 1.2
    for j in range(10):
        cc = nest.GetConnections(target=[id])
        weight = nest.GetStatus(cc, ['source', 'weight'])
        length = len(nest.GetStatus(cc, ['weight']))
        line = []
        for i in range(len(weight)):
            line.append(list(weight[i]))
        line.sort(key=lambda x: x[1])
        for i in range(len(line)):
            # abs(line[i][1]-1.0) <= change  ===> line[i][1]<=
            if abs(line[i][1] - 1.0) <= change and (min(self.__predictLayer['inputLayer']) <= line[i][0] < max(self.__predictLayer['inputLayer'])):
                nest.Disconnect(pre=[line[i][0]], post=[id], conn_spec={'rule': "one_to_one"},syn_spec={'model': 'static_synapse'})
                length -= 1
            id += 1

if __name__ == '__main__':

    print(alpha(1-0.872))
    print(np.e ** alpha(1-0.872))
    print(np.e ** -alpha(1-0.872))
