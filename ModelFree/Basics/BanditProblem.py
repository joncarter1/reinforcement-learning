import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy as copy

def ucb(t, x):
    return ((2*np.log(t))/x)**0.5

import matplotlib as mpl
plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class GaussianBandit:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.qs = []

    def draw(self):
        return np.random.normal(loc=self.mu, scale=self.sigma)

    def plot(self, plot_ind, iterations=None):
        if not iterations:
            iterations = len(self.qs)
        #y = np.linspace(self.mu - 3*self.sigma, self.mu + 3*self.sigma, 100)
        #plt.plot((iterations/5)*stats.norm.pdf(y, self.mu, self.sigma), y,color=plot_colors[plot_ind])
        plt.plot(np.linspace(0, iterations, len(self.qs)), self.qs, label=f'$\mu$ = {self.mu}',color=plot_colors[plot_ind])


class kArmedBandit:
    def __init__(self, bandits):
        self.bandits = bandits
        self.k = len(bandits)
        self.plot_ind = 0

    def draw(self, i):
        if 0 <= i < self.k:
            return self.bandits[i].draw()
        else:
            raise IndexError('Bandit index out of range')

    def update_qs(self, q_table):
        for i in range(self.k):
            self.bandits[i].qs.append(q_table[i])

    def plot_bandits(self, iterations=None):
        for bandit in self.bandits:
            bandit.plot(self.plot_ind, iterations)
            self.plot_ind += 1
        plt.xlabel('Iterations')
        plt.ylabel('Expected bandit reward')



gb1 = GaussianBandit(1, 2)
gb2 = GaussianBandit(0, 0.1)
gb3= GaussianBandit(1.5, 1)

kbandit1 = kArmedBandit([gb1, gb2, gb3])

e = 0.3


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

np.random.seed(1)
total_reward = 0
cumulative_rewards = []

class GreedyPolicy:
    def __init__(self, kbandit, epsilon):
        self.iterations = 0
        self.kbandit = kbandit
        self.q_table = np.zeros(shape=(kbandit.k))
        self.count_table = np.ones(shape=(kbandit.k))
        self.reward = 0
        self.total_reward = 0
        self.cumulative_rewards =[]
        self.e = epsilon

    def take_action(self):
        if np.random.random() > self.e:
            action = np.argmax(self.q_table)
        else:
            action = np.random.randint(0, self.kbandit.k)
        self.reward = self.kbandit.draw(action)
        self.total_reward += self.reward
        self.cumulative_rewards.append(self.total_reward)
        self.count_table[action] += 1
        self.q_table[action] = self.q_table[action] + (1 / self.count_table[action]) * (self.reward - self.q_table[action])
        if self.iterations % 50 == 0:
            self.kbandit.update_qs(self.q_table)
        self.iterations += 1
        self.e *= 0.99
        self.e = max(0.1, self.e)

class SoftmaxPolicy:
    def __init__(self, kbandit):
        self.iterations = 0
        self.kbandit = kbandit
        self.q_table = np.zeros(shape=(kbandit.k))
        self.count_table = np.ones(shape=(kbandit.k))
        self.reward = 0
        self.total_reward = 0
        self.cumulative_rewards =[]
    def take_action(self):
        action = np.random.choice(self.kbandit.k, p=softmax(self.q_table))
        self.reward = self.kbandit.draw(action)
        self.total_reward += self.reward
        self.cumulative_rewards.append(self.total_reward)
        self.count_table[action] += 1
        self.q_table[action] = self.q_table[action] + (1 / self.count_table[action]) * (self.reward - self.q_table[action])
        if self.iterations % 50 == 0:
            self.kbandit.update_qs(self.q_table)
        self.iterations += 1

class UCBPolicy:
    def __init__(self, kbandit):
        self.iterations = 0
        self.kbandit = kbandit
        self.q_table = np.zeros(shape=(kbandit.k))
        self.count_table = np.ones(shape=(kbandit.k))
        self.reward = 0
        self.total_reward = 0
        self.cumulative_rewards =[]
    def take_action(self):
        action = np.argmax(self.q_table+ucb(1+self.iterations, self.count_table))
        self.reward = self.kbandit.draw(action)
        self.total_reward += self.reward
        self.cumulative_rewards.append(self.total_reward)
        self.count_table[action] += 1
        self.q_table[action] = self.q_table[action] + (1 / self.count_table[action]) * (self.reward - self.q_table[action])
        if self.iterations % 50 == 0:
            self.kbandit.update_qs(self.q_table)
        self.iterations += 1

softpolicy = SoftmaxPolicy(kbandit1)
greedypolicy = GreedyPolicy(copy(kbandit1), 0.1)
ucbpolicy = UCBPolicy(copy(kbandit1))

iterations = 10000
for iteration in range(iterations):
    softpolicy.take_action()
    greedypolicy.take_action()
    ucbpolicy.take_action()

plt.figure(1)
softpolicy.kbandit.plot_bandits(iterations=iterations)
plt.title('Softmax')
plt.legend()
plt.figure(2)
greedypolicy.kbandit.plot_bandits(iterations=iterations)
plt.title('Greedy')
plt.legend()
plt.figure(3)
ucbpolicy.kbandit.plot_bandits(iterations=iterations)
plt.title('UCB1')
plt.legend()
plt.show()
#plt.savefig('Cumulative')

#plt.plot(softpolicy.cumulative_rewards, label='Softmax') #.bandits[0].qs)
#plt.plot(greedypolicy.cumulative_rewards, label='e-greedy')#.bandits[0].qs)
#plt.plot(ucbpolicy.cumulative_rewards,label='UCB1')#.bandits[0].qs)
