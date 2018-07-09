import gym
import numpy as np
import math

from random import random, sample


class Ninja:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

        self.levels = [0, 0, 5, 2]

        self.Q = self.enumerate_state_space()

        self.epsilon = 0.9
        self.alpha = 0.9
        self.gamma = 0.95

        self.high = [0.3, 2.25, 0.4, 3.5]
        self.low = [-0.3, -2.25, -0.4, -3.5]

        self.max_iters = 100
        self.max_time = 200

        self.n_space = self.env.observation_space.shape[0]

    def run(self):
        for i_episode in range(self.max_iters):
            observation = self.env.reset()

            current_state = self.discretize(observation)
            action = self.get_action(current_state)

            total_reward = 0

            data_t = []
            data_r = []

            for t in range(self.max_time):
                observation, reward, done, _ = self.env.step(action)

                new_state = self.discretize(observation)
                action = self.get_action(current_state)

                self.update_Q(current_state, new_state, action)

                current_state = new_state

                total_reward += reward

                if done:
                    data_t.append(t)
                    data_r.append(total_reward)

                    #if (i_episode + 1) % 2 == 0 or True:
                    if (i_episode + 1) % 500 == 0:
                        #print("%6d %3d %3d" % (i_episode + 1, np.mean(data_t), np.mean(data_r)))
                        #print("%6d %3d %3d %6.3f %6.3f %6.3f" %
                                #(i_episode + 1, np.mean(data_t), np.mean(data_r), epsilon, alpha, gamma))
                        data_t = []
                        data_r = []
                    break

    def update_Q(self):
        Q[current_state][action] += alpha * (reward + \
                gamma * max_Q_value(Q, new_state) - get_Q(Q, current_state, action))

    def discretize(self, space):
        v = [0 for _ in range(self.n_space)]

        for i in range(self.n_space):
            v[i] = int(round(((space[i] - self.low[i]) / (self.high[i] -
                       self.low[i])) * self.levels[i]))

        return tuple(v)

    def get_action(self, state):
        if random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def enumerate_state_space(self):
        actions = self.env.action_space.n
        Q = {}

        counter = [0 for _ in range(len(self.levels))]
        do = True
        while do:
            for i in range(len(self.levels) - 1, -1, -1):
                if counter[i] < self.levels[i]:
                    counter[i] += 1
                    break
                else:
                    if i > 0:
                        counter[i] = 0
                    elif i == 0:
                        do = False
                        break

            if not do:
                break

            s = (tuple(counter))

            Q[s] = {}
            for a in range(actions):
                Q[s][a] = 0.0

        return Q
