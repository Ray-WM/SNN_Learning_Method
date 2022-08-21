import rl_network
import numpy as np
import pandas as pd

class QLtable(object):
    def __init(self, actions, learning_rate=0.01, reward_decay=0.8, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 落入区间内，选择动作
        if np.random.uniform < self.epsilon:
            # 选择最佳动作
            state_action = self.q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if abs(s_[3] - 1) < 1e-5 & abs(s_[1] - s_[3]) < 1e-5:
            q_target = r        #达到停止条件
        else:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 导入一个新的状态
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

