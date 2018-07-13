import agent
import numpy as np
import sys


class Searcher:
    def __init__(self):
        self.champion = agent.Ninja()
        self.challenger = agent.Ninja()

        self.champion.random_params()
        self.challenger.random_params()

        self.matches = 10
        self.generations = 10

    def step(self, warmup=False):
        self.challenger.mutate()

        for i in range(self.matches):
            self.challenger.run()

            if warmup:
                self.champion.run()

            print('.', end='')
            sys.stdout.flush()

        print()

        self.compare(replace=warmup)
        self.challenger.reset()

    def run(self):
        self.step(warmup=True)

        for gen in range(self.generations):
            self.step()

        self.champion.print_params()

    def compare(self, replace=False):
        m1 = np.mean(self.champion.max_score)
        m2 = np.mean(self.challenger.max_score)

        print("Champion:%6.3f   Challenger: %6.3f" % (m1, m2))

        if m2 > m1:
            self.swap()

        if replace:
            self.challenger.alpha = self.champion.alpha
            self.challenger.gamma = self.champion.gamma
            self.challenger.epsilon = self.champion.epsilon
            self.challenger.levels = self.champion.levels.copy()

    def swap(self):
        a = self.champion
        self.champion = self.challenger
        self.challenger = a

    def reset_challenger(self):
        self.challenger.reset()


if __name__ == '__main__':
    searcher = Searcher()
    searcher.run()
