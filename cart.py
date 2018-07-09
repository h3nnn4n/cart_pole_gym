import gym
import numpy as np
import math

from random import random, sample


def max_Q(Q, state):
    if state not in Q.keys():
        return sample([0, 1], k=1)[0]

    v = list(Q.values())
    k = list(Q.keys())

    m = max(v)
    c = []
    for i, j in enumerate(v):
        if j == m:
            c.append(i)

    if len(c) == 1:
        return k[v.index(m)]
    else:
        return sample(c, 1)[0]


def max_Q_value(Q, state):
    a = max_Q(Q, state)
    if state not in Q.keys() or a not in Q[state].keys():
        return 0
    return Q[state][a]


def discretize(space, levels, h, l):
    n_space = env.observation_space.shape[0]
    v = [0 for _ in range(n_space)]

    for i in range(n_space):
        v[i] = int(round(((space[i] - l[i]) / (h[i] - l[i])) * levels[i]))

    return tuple(v)


def get_action(Q, state, env, epsilon):
    max_actions = env.action_space.n
    if state in Q.keys() and random() < epsilon:
        if len(Q[state]) > max_actions:
            return max_Q(Q[state])
    return env.action_space.sample()


def get_Q(Q, state, action):
    if state in Q.keys():
        if action in Q[state].keys():
            return Q[state][action]
    return 0


def update_Q(Q, current_state, new_state, action, alpha, gamma):
    if current_state not in Q.keys():
        Q[current_state] = {}

    if current_state in Q.keys():
        if action not in Q[current_state].keys():
            Q[current_state][action] = 0
    #print(max_Q_value(Q, new_state))
    #print(get_Q(Q, new_state, action))
    Q[current_state][action] += alpha * (reward + \
            gamma * max_Q_value(Q, new_state) - get_Q(Q, current_state, action))


env = gym.make('CartPole-v0')

levels = [0, 0, 5, 2]

#Q = enumerate_state_space(levels, env)
Q = {}

epsilon = 0.9
alpha = 0.9
gamma = 0.95

decay_1 = 0.9999
decay_2 = 0.9995

n_space = env.observation_space.shape[0]
h = [float('-inf') for _ in range(n_space)]
l = [float('inf') for _ in range(n_space)]

#h = [0.27381544672682456, 2.196749673236612, 0.05094564389112599, 0.04999926542898585]
#l = [-0.05093429589386663, -0.04999975830341763, -0.26919109066233127, -3.336362821304928]

h = [0.3, 2.25, 0.4, 3.5]
l = [-0.3, -2.25, -0.4, -3.5]

for i_episode in range(10000):
    observation = env.reset()

    current_state = discretize(observation, levels, h, l)
    action = get_action(Q, current_state, env, epsilon)

    total_reward = 0

    data_t = []
    data_r = []

    epsilon *= decay_1
    alpha *= decay_2
    epsilon *= decay_2

    for t in range(200):
        observation, reward, done, _ = env.step(action)

        new_state = discretize(observation, levels, h, l)
        action = get_action(Q, current_state, env, epsilon)

        update_Q(Q, current_state, new_state, action, alpha, gamma)

        current_state = new_state

        for i in range(n_space):
            h[i] = max(h[i], observation[i])
            l[i] = min(l[i], observation[i])

        total_reward += reward

        if done:
            data_t.append(t)
            data_r.append(total_reward)

            #if (i_episode + 1) % 2 == 0 or True:
            if (i_episode + 1) % 500 == 0:
                #print("%6d %3d %3d" % (i_episode + 1, np.mean(data_t), np.mean(data_r)))
                print("%6d %3d %3d %6.3f %6.3f %6.3f" %
                        (i_episode + 1, np.mean(data_t), np.mean(data_r), epsilon, alpha, gamma))
                data_t = []
                data_r = []
            break
