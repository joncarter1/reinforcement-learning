import numpy as np


def softmax(x):
    return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))


def greedy_choice(x, epsilon):
    return


class GreedyPolicy:
    def __init__(self, epsilon=1, decay=0.9995):
        self.e_initial = epsilon
        self.e_decay = decay
        self.parameter = self.e_initial

    def choose_action(self, q_values, N_values, iterations):
        log_parameter = np.log(self.e_initial)+iterations*np.log(self.e_decay)
        if log_parameter < -500:  # Prevent any weird behaviour due to numerical instability.
            return np.argmax(q_values)
        self.parameter = self.e_initial*self.e_decay**iterations
        if np.random.random() > self.parameter:
            return np.argmax(q_values)
        else:
            return np.random.randint(0, len(q_values))

class SoftmaxPolicy:
    def __init__(self, tau=1, tau_increase=1.01):
        self.tau_initial = tau
        self.tau_increase = tau_increase
        self.parameter = tau

    def choose_action(self, q_values, N_values, iterations):
        log_parameter = np.log(self.tau_initial)+iterations*np.log(self.tau_increase)
        if log_parameter > 40:  # Basically deterministic
            return np.argmax(q_values)
        self.parameter = self.tau_initial * self.tau_increase ** iterations - 1
        return np.random.choice(len(q_values), p=softmax(self.parameter*q_values))

def ucb(t, x):
    return ((2*np.log(t))/x)**0.5

class UCBPolicy:
    def __init__(self, c=1):
        self.c = c

    def choose_action(self, q_values, N_values, iterations):
        return np.argmax(q_values+self.c*ucb(1+iterations, N_values))