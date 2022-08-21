import imp
from whatnet import q_learning
from whatnet import rl_network
import rl_train
import pandas as pd
from whatnet.data.cnn_converter import CnnConverter

def update(RL:q_learning):
    # 记录行为
    df = pd.DataFrame(columns=('state','action_space','reward','Q','action'))


    for i in range(576):
        for j in range(10):
            # 初始化状态
            observation = [1,0,1,0]
            action = RL.QLtable.choose_action(observation)

            s_new, reward = rl_train.Env.step(action, i, j ,observation)

            RL.QLtable.learn(observation, action, reward, s_new)
            q = RL.QLtable.q_table.loc[observation, action]
            df = df.append(pd.DataFrame({'state':[observation],'action_space':[env.action_space[action]],'reward':[reward],'Q':[q],'action':action}), ignore_index=True)
            
            observation = s_new

    #---------------------------------
    df.to_csv('action.csv')
    RL.QLtable.q_table.to_csv('q_table.csv')


if __name__=="__main__":
    path = "weight.csv"
    cnn = CnnConverter(param_file_path="./config/converter_config_no_convolution.json")
    rl_train(path, cnn)
    RL = q_learning.QLtable(action=list(range(rl_train.Env.n_action)))



