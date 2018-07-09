import gym
import numpy as np
import math

from random import random, sample


def max_Q(Q):
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
        #w = sample(c, 1)[0]
        #print('haha ', w)
        #return(w)
        return sample(c, 1)[0]


def discretize(space, levels, h, l):
    n_space = env.observation_space.shape[0]
    v = [0 for _ in range(n_space)]

    for i in range(n_space):
        v[i] = int(round(((space[i] - l[i]) / (h[i] - l[i])) * levels[i]))

    return tuple(v)


def enumerate_state_space(levels, env):
    actions = env.action_space.n
    Q = {}

    counter = [0 for _ in range(len(levels))]
    do = True
    while do:
        for i in range(len(levels) - 1, -1, -1):
            if counter[i] < levels[i]:
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
            #Q[s][a] = random() / 100.0
            Q[s][a] = 0.0

    return Q


env = gym.make('CartPole-v0')

levels = [0, 0, 15, 20]

Q = enumerate_state_space(levels, env)

epsilon = 0.6
alpha = 0.9
gamma = 0.95

decay_1 = 0.99
decay_2 = 0.95

n_space = env.observation_space.shape[0]
h = [float('-inf') for _ in range(n_space)]
l = [float('inf') for _ in range(n_space)]

#h = [0.27381544672682456, 2.196749673236612, 0.05094564389112599, 0.04999926542898585]
#l = [-0.05093429589386663, -0.04999975830341763, -0.26919109066233127, -3.336362821304928]

h = [0.3, 2.25, 0.4, 3.5]
l = [-0.3, -2.25, -0.4, -3.5]

#for i_episode in range(5000):
#for i_episode in range(500):
for i_episode in range(50):
    observation = env.reset()

    current_state = discretize(observation, levels, h, l)
    possible_actions = Q[current_state]
    action = max_Q(possible_actions)

    total_reward = 0

    #print()

    data_t = []
    data_r = []

    epsilon *= decay_1
    alpha *= decay_2
    epsilon *= decay_2

    for t in range(200):
        #if i_episode == 1500:
            #env.render()

        observation, reward, done, _ = env.step(action)

        new_state = discretize(observation, levels, h, l)
        possible_actions = Q[new_state]

        if random() < epsilon:
            action = env.action_space.sample()
        else:
            action = max_Q(possible_actions)

        #Q[current_state][action] += alpha * ((1.0 - reward * min(1, abs(observation[0]))) + gamma * max_Q(Q[new_state]) - Q[current_state][action])
        Q[current_state][action] += alpha * (reward + gamma * max_Q(Q[new_state]) - Q[current_state][action])
        current_state = new_state

        #print(action)

        for i in range(n_space):
            h[i] = max(h[i], observation[i])
            l[i] = min(l[i], observation[i])

        total_reward += reward

        if done:
            data_t.append(t)
            data_r.append(total_reward)

            if (i_episode + 1) % 2 == 0 or True:
                print("%6d %3d %3d" % (i_episode + 1, np.mean(data_t), np.mean(data_r)))
                data_t = []
                data_r = []
            break

#for k, v in Q.items():
    #if max_Q(Q[k]) > 0:
        #print(k, v)
