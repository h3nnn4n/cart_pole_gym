import gym
import numpy as np
import sys
from random import random, randint, sample


class Ninja:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

        self.levels = [0, 0, 2, 2]

        self.Q = self.enumerate_state_space()

        self.alpha = 0.4
        self.gamma = 0.9
        self.epsilon = 0.75

        self.alpha_min = 0.6
        self.gamma_min = 0.5
        self.epsilon_min = 0.05

        self.high = [10.5, 7.0, 6.0, 10.0]
        self.low = [-10.5, -7.0, -6.0, -10.0]

        self.max_iters = 1000
        self.max_iters = 2000
        self.max_iters = 10000
        self.log_interval = 100
        self.max_time = 200

        self.log = True

        self.n_space = self.env.observation_space.shape[0]

        self.max_score = []
        self.mean_score = []

    def print_params(self):
        print("alpha: %6.3f   gamma: %6.3f   epsilon: %6.3f   levels: %s" %
              (self.alpha, self.gamma, self.epsilon, self.levels))

    def random_params(self):
        self.alpha = random()
        self.gamma = random()
        self.epsilon = random()

        self.levels[2] = randint(5, 11)
        self.levels[3] = randint(5, 11)

    def mutate(self):
        redo = True

        while redo:
            if False:
                r = randint(0, 4)
                if r == 0:
                    self.alpha += random() / 5 - 0.2
                elif r == 1:
                    self.gamma += random() / 5 - 0.2
                elif r == 2:
                    self.epsilon += random() / 5 - 0.2
                elif r == 3:
                    self.levels[2] += -1 if random() < 0.5 else 1
                elif r == 4:
                    self.levels[3] += -1 if random() < 0.5 else 1
            else:
                self.alpha += random() / 5 - 0.2
                self.gamma += random() / 5 - 0.2
                self.epsilon += random() / 5 - 0.2
                self.levels[2] += -randint(1, 3) if random() < 0.5 else \
                    randint(1, 3)
                self.levels[3] += -randint(1, 3) if random() < 0.5 else \
                    randint(1, 3)

            if self.alpha > 1:
                self.alpha = 1

            if self.gamma > 1:
                self.gamma = 1

            if self.epsilon > 0.6:
                self.epsilon = 0.6

            if self.alpha < 0.1:
                self.alpha = 0.1

            if self.gamma < 0.1:
                self.gamma = 0.1

            if self.epsilon < 0.1:
                self.epsilon = 0.1

            if self.levels[2] < 1:
                self.levels[2] = 1

            if self.levels[3] < 1:
                self.levels[3] = 1

            redo = False

    def get_alpha(self):
        return self.alpha

    def get_gamma(self):
        return self.gamma

    def get_epsilon(self):
        return self.epsilon

    def update_params(self):
        self.alpha *= 1.0001
        self.gamma *= 0.9999
        self.epsilon *= 0.99975

        self.alpha = min(self.alpha, self.alpha_min)
        self.gamma = max(self.gamma, self.gamma_min)
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def reset_Q(self):
        self.Q = self.enumerate_state_space()

    def reset(self):
        self.Q = self.enumerate_state_space()
        self.max_score = []
        self.mean_score = []

    def run(self, reset_Q=True):
        if reset_Q:
            self.reset_Q()

        data_t = []
        data_r = []

        for i_episode in range(self.max_iters):
            self.update_params()

            observation = self.env.reset()

            current_state = self.discretize(observation)
            action = self.get_action(current_state)

            total_reward = 0

            for t in range(self.max_time):
                #if i_episode > 2000:
                    #self.env.render()

                observation, reward, done, _ = self.env.step(action)

                new_state = self.discretize(observation)
                action = self.get_action(current_state)

                self.update_Q(current_state, new_state, action, reward)

                current_state = new_state

                total_reward += reward

                if done:
                    data_t.append(t)
                    data_r.append(total_reward)

                    if (i_episode + 1) % self.log_interval == 0:
                        if self.log and len(data_r):
                            print("%6d %3d %3d %3d %6.3f %6.3f %6.3f" %
                                  (i_episode + 1, np.mean(data_t),
                                   max(data_r), total_reward,
                                   self.alpha, self.gamma, self.epsilon))
                            sys.stdout.flush()

                        data_t = []
                        data_r = []
                    break

    def update_Q(self, current_state, new_state, action, reward):
        self.Q[current_state][action] += self.get_alpha() * \
                (reward + self.get_gamma() *
                 np.argmax(self.Q[new_state]) - self.Q[current_state][action])

    def discretize(self, space):
        v = [0 for _ in range(self.n_space)]

        for i in range(self.n_space):
            if space[i] < self.low[i]:
                print('New low found: %2d %6.3f' % (i, space[i]))

            if space[i] > self.high[i]:
                print('New high found: %2d %6.3f' % (i, space[i]))

            v[i] = int(round(((space[i] - self.low[i]) / (self.high[i] -
                       self.low[i])) * self.levels[i]))

        return tuple(v)

    def get_action(self, state):
        if random() < self.get_epsilon():
            return self.env.action_space.sample()
        else:
            best_index = np.argmax(self.Q[state])
            best_reward = self.Q[state][best_index]
            bests_indices = [action for action in self.Q[state].keys() if self.Q[state][action] == best_reward]
            return sample(bests_indices, k=1)[0]

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


if __name__ == '__main__':
    ninja_das_trevas = Ninja()
    ninja_das_trevas.run()
