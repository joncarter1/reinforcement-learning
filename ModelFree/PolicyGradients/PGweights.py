import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from ModelFree.PolicyGradients.REINFORCE import PolicyEstimator

inds = [(11, 13), (4, 13), (11, 2), (5, 8)]


def plot_stats(model_name):
    #agent = PolicyEstimator('LunarLander-v2', model_name)
    weights1 = pk.load(open(f"{model_name}/weights1.p","rb"))
    weights2 = pk.load(open(f"{model_name}/weights2.p", "rb"))
    scores = pk.load(open(f"{model_name}/scores.p", "rb"))

    plt.figure(1)
    for ind in inds:
        plt.plot([weight[ind] for weight in weights1], label = str(ind))

    plt.xlabel("Episode")
    plt.ylabel("Weights")
    plt.legend(title="Weight Matrix index")
    plt.figure(2)
    plt.plot(scores)
    N = 100
    plt.plot(np.convolve(scores, np.ones((N,))/N, mode='same'))
    plt.xlabel("Episode")
    plt.ylabel("Scores")
    plt.show()

if __name__ == '__main__':
    plot_stats('LunarLander-v2/Discount0.95')
    plot_stats('LunarLander-v2/Baseline0.99')