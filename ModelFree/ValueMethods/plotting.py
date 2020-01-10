import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import array as arr

def data(strategy, l, discount, parameter):
    return pickle.load(open(f"results/{strategy}/{l}-{discount}-{parameter}.p", "rb"))

if __name__ == "__main__":
    strategy, l, gamma, parameter = 'softmax', 0.2, 0.97, 1.01

    strategies = ['greedy','softmax']
    alphas = [r'$\epsilon$', r'1/$\tau$']
    parameters = [[0.99,0.995,0.999],[1.0005, 1.001, 1.01]]
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for ax1, strategy, alpha, params in zip(axes, strategies, alphas, parameters):
        ax1.set_xlim(0,20000)
        ax1.set_ylim(-200, -120)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='y')
        #ax1.scatter(np.arange(0, 10001), ep_rewards, marker='x')
        #params = [1.001, 1.01, 1.05]
        for index, param in enumerate(params):
            ep_rewards, aggr_ep_rewards, agent = data(strategy, l, gamma, param)
            #ax1.scatter(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label=f'max {param}', marker='x')
            #ax1.scatter(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label=f'min {param}', marker='x')
            N = 10
            smoothed_rewards = np.convolve(ep_rewards, np.ones((N,))/N, mode='same')
            strategy_name = r"$\epsilon$-"+strategy if strategy == 'greedy' else strategy
            ax1.plot(np.arange(N, len(smoothed_rewards)- N), smoothed_rewards[N:-N], label=rf'$\alpha$ = {param}')
            ax1.set_title(f"Smoothed rewards using \nthe {strategy_name} policy", size=16)
            """if index == 0: ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel(alpha)
            if strategy == 'greedy':
                ax2.plot(np.arange(10001), param**np.arange(10001), label=rf'$\alpha$ = {param}')
            else:
                ax2.plot(np.arange(10001), 1 / param ** np.arange(10001), label=rf'$\alpha$ = {param}')

            ax2.tick_params(axis='y')"""

        ax1.legend()


    fig.tight_layout()
    plt.savefig("Policies")
    plt.show()
