import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.layers import Dropout, Activation, Dense, Input, GaussianNoise
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import pickle
import random
import tensorflow as tf
import keras.backend as K
import numpy as np

early_stopping = EarlyStopping(monitor='val_loss', patience=1)

def gaussian_nll(ytrue, ypreds):
    """
    Modified code originally written by Sergey Prokudin
    Accessed: Sat 2nd May 2020
    URL: https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
    Keras implementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, :n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5 * K.sum(K.square((ytrue - mu) / K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)
    logsigma_min, logsigma_max = -12, 1
    log_likelihood = mse + sigma_trace + log2pi
    #sigma_penalty = K.sum(100*(tf.math.exp(10*(logsigma - logsigma_max)) + tf.math.exp(10*(logsigma_min - logsigma))), axis=1)
    return K.mean(-log_likelihood)


def create_model(input_size, output_size,
                 probabilistic=True,
                 layers=2, neurons=500,
                 output_activation="linear",
                 dr=0.0, noise=0,
                 l2_penalty=0.0, lr=0.001):
    loss = "mse"
    if probabilistic:
        loss, output_size = gaussian_nll, 2 * output_size

    input_layer = Input(shape=(input_size,))
    if noise:
        current_layer = GaussianNoise(stddev=noise)(input_layer)
    else:
        current_layer = input_layer

    for i in range(layers):
        dense_layer = Dense(neurons, kernel_initializer='random_normal', bias_initializer='zeros',
                            kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty),
                            activation="relu")(current_layer)
        current_layer = Dropout(dr)(dense_layer, training=True) if dr else dense_layer
    output_layer = Dense(output_size, activation=output_activation)(current_layer)
    model_nn = Model(input=[input_layer], output=[output_layer])
    optimizer = Adam(learning_rate=lr)
    model_nn.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    #plot_model(model_nn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model_nn


class NNModel:
    def __init__(self, input_size, output_size,
                 probabilistic=True,
                 b=1, output_activation="linear",
                 layers=2, neurons=500,
                 dr=0, l2_penalty=0, lr=0.001,
                 data_stats=[0, 1, 0, 1]):
        self.B=b
        self.probabilistic = probabilistic
        self.nns = [create_model(input_size, output_size,
                                 probabilistic=probabilistic,
                                 layers=layers, neurons=neurons,
                                 dr=dr, l2_penalty=l2_penalty, lr=lr,
                                 output_activation=output_activation) for _ in range(self.B)]
        self.input_mean, self.input_std, self.output_mean, self.output_std = data_stats
        self.min_vars = [0 for _ in range(self.B)]
        self.max_vars = [100 for _ in range(self.B)]

    def train(self, x, y, b_i=1, batch_size=512, verbose=1, epochs=1, validation_split=0.1, shuffle=True):
        """Train i'th neural network of model on normalised targets.
            Small amount of noise added to help prevent mode collapse of output distribution."""
        normalised_x = (x - self.input_mean)/self.input_std
        normalised_y = (y-self.output_mean)/self.output_std
        #noisy_x = normalised_x + np.random.normal(0, 1e-3, size=x.shape)
        #noisy_y = normalised_y + np.random.normal(0, 1e-3, size=y.shape)
        self.nns[b_i].fit(x=normalised_x, y=normalised_y, batch_size=batch_size, verbose=verbose,
                          epochs=epochs, validation_split=validation_split, shuffle=shuffle, callbacks=[early_stopping])
        return

    def predict(self, x, b_i=1):
        """Predict output using i'th neural network of bootstrap."""
        normalised_input = (x-self.input_mean)/self.input_std
        normalised_output = self.nns[b_i].predict(x=normalised_input)
        y_out = normalised_output*self.output_std + self.output_mean
        return y_out

    def save(self, name):
        pickle.dump(self, open(name, "wb"))


def main(EPOCHS):
    input_size, output_size = 12, 10
    state_action_buffer = pickle.load(open("model_data", "rb"))
    state_actions = state_action_buffer[:-1]
    next_states = state_action_buffer[1:, :10]
    delta_states = next_states - state_actions[:, :10]
    mu_in, std_in = np.mean(state_actions, axis=0), np.std(state_actions, axis=0)
    mu_out, std_out = np.mean(delta_states, axis=0), np.std(delta_states, axis=0)
    data_stats = mu_in, std_in, mu_out, std_out
    nn_policy = NNModel(input_size, output_size, data_stats=data_stats)
    nn_policy.train(state_actions, delta_states, EPOCHS)
    nn_policy.save("new_model")
    return



if __name__ == "__main__":
    """main(2)
    new_nn = pickle.load(open("new_model","rb"))
    state_action_buffer = pickle.load(open("model_data", "rb"))"""
    x = np.linspace(-0.1, 1)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, soft_penalty(x))
    plt.show()

