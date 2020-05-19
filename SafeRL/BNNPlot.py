import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
from handful_of_trials.dmbrl.modeling.layers import FC
from handful_of_trials.dmbrl.modeling.models import BNN
from dotmap import DotMap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from SafeRL.MPC import MPCLearner
pe_params = DotMap(name="model", num_networks=5, sess=None, load_model=None, model_dir="PETS")

pe_model = BNN(pe_params)

state_dims = 10
action_dims = 2
input_dims, output_dims = 1, 1

pe_model.add(FC(200, input_dim=input_dims, activation="swish", weight_decay=0.000025))
pe_model.add(FC(200, activation="swish", weight_decay=0.00005))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
#pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(output_dim=output_dims, weight_decay=0.0001))
pe_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
#pe_model.finalize(tf.train.RMSPropOptimizer, {"learning_rate": 0.001})

t = np.hstack((np.linspace(-4, -1, 100), np.linspace(1, 4, 150)))


x = np.tanh(t)*np.sin(t)

noise_var = np.clip(0.1*np.sin(t), 0.005, None)
noisy_x = x + np.random.multivariate_normal(mean=np.zeros(len(x)), cov=np.diag(noise_var))

t = t[:, np.newaxis]
noisy_x = noisy_x[:, np.newaxis]

t_all = np.linspace(-10, 10, 1000)[:, np.newaxis]

if __name__ == "__main__":
    plt.figure()
    plt.scatter(t, noisy_x, label="Data")
    pe_model.train(inputs=t, targets=noisy_x, epochs=100, holdout_ratio=0.1)
    means, vars = pe_model.predict(inputs=t_all, factored=True)
    total_mean = np.squeeze(np.mean(means, axis=0))
    total_std = np.squeeze(np.std(means, axis=0))
    print(total_mean.shape)

    t_all = np.squeeze(t_all)

    for mean, var in zip(means, vars):
        mean, var = np.squeeze(mean), np.squeeze(var)
        plt.plot(t_all, mean, "--", linewidth=0.2, color="red")
        plt.fill_between(t_all, mean-var, mean+var, color="green", alpha=0.1)

    plt.plot(t_all, total_mean, color="green", label="Ensemble mean")
    plt.fill_between(t_all, total_mean - 3*total_std, total_mean + 3*total_std, alpha=0.1, color="green", label="Ensemble variance")
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()