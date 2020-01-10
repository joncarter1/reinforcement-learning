import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from numpy import array as arr
inds = [(11, 8), (4, 13), (11, 2), (5, 8)]


def plot_stats(model_name, label):
    #agent = PolicyEstimator('LunarLander-v2', model_name)
    #weights1 = pk.load(open(f"{model_name}/weights1.p","rb"))
    #weights2 = pk.load(open(f"{model_name}/weights2.p", "rb"))
    scores = arr(pk.load(open(f"{model_name}/scores.p", "rb")))
    """
    plt.figure(1)
    for ind in inds:
        plt.plot([weight[ind] for weight in weights1], label = str(ind))
    plt.xlabel("Episode")
    plt.ylabel("Weights")
    plt.legend(title="Weight Matrix index")"""
    plt.figure(2)
    #plt.plot(scores)
    N = 100
    min_score = -100 - 0.3 * 1000
    print(min_score)
    scores[np.where(scores < min_score)] = min_score
    #label = model_name.split("/")[-1]
    plt.xlim(0, 2000)
    #plt.plot(np.arange(N, len(scores)-N), np.convolve(scores, np.ones((N,))/N, mode='same'), label=label)
    smoothed_scores = np.convolve(scores, np.ones((N,)) / N, mode='same')
    smoothed_stds = np.convolve((scores-smoothed_scores)**2, np.ones((N,)) / N, mode='same')**0.5
    plot_range = np.arange(N, len(scores)-N)
    plt.plot(plot_range, smoothed_scores[N:-N], label=label)
    plt.fill_between(plot_range, (smoothed_scores-smoothed_stds)[N:-N], (smoothed_scores+smoothed_stds)[N:-N], alpha=0.3)

    plt.xlabel("Episode")
    plt.ylabel("Scores")


if __name__ == '__main__':
    #plot_stats('LunarLander-v2/NB0.99')
    #plot_stats('LunarLander-v2/B0.99')
    #plot_stats('LunarLander-v2/BM0.99')
    plot_stats('LunarLander-v2/G0.972', label=r"$\gamma=0.97$")
    plot_stats('LunarLander-v2/G0.993', label=r"$\gamma=0.99$")


    plt.legend()
    #plt.savefig("baselines")
    plt.show()