import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.layers import Dropout, Activation, Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import pickle


def create_model(input_size,
                 output_size,
                 layers=2,
                 neurons=500,
                 output_activation="linear",
                 dr=0.0,
                 l2_penalty=0.0,
                 lr=0.001):

    input_layer = Input(shape=(input_size,))
    current_layer = input_layer
    for i in range(layers):
        dense_layer = Dense(neurons, kernel_regularizer=l2(l2_penalty),
                       bias_regularizer=l2(l2_penalty),
                       activation="relu")(current_layer)
        dropout_layer = Dropout(dr)(dense_layer, training=True)
        current_layer = dropout_layer
    output_layer = Dense(output_size, activation=output_activation)(current_layer)
    model_nn = Model(input=[input_layer], output=[output_layer])
    optimizer = Adam(learning_rate=lr)
    model_nn.compile(optimizer=optimizer, loss="mse", metrics=['accuracy'])
    #plot_model(model_nn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model_nn


class NNModel:
    def __init__(self, input_size,
                 output_size,
                 output_activation="linear",
                 data_stats=[0,1,0,1]):
        self.nn = create_model(input_size,
                               output_size)
        self.input_mean, self.input_std, self.output_mean, self.output_std = data_stats

    def train(self, x, y, batch_size=512):
        normalised_x = (x - self.input_mean)/self.input_std
        normalised_y = (y-self.output_mean)/self.output_std
        #gaussian_noise = np.random.multivariate_normal(0, 0.001)
        #noisy_x = normalised_x + gaussian_noise
        self.nn.fit(x=normalised_x, y=normalised_y, batch_size=batch_size, verbose=1, shuffle=True)
        return

    def predict(self, x):
        normalised_input = (x-self.input_mean)/self.input_std
        normalised_output = self.nn.predict(x=normalised_input)
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
    main(2)
    new_nn = pickle.load(open("new_model","rb"))
    state_action_buffer = pickle.load(open("model_data", "rb"))

