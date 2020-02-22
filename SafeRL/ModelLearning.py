import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.layers import Dropout, Activation, Dense, Input
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import pickle

def create_model(input_size, output_size):
    #lr = 0.01
    dr = 0.0
    l2_penalty = 0.0
    l1_dims, l2_dims, l3_dims, l4_dims = 300, 300, 300, 300
    input_layer = Input(shape=(input_size,))
    dense1 = Dense(l1_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(input_layer)
    dropout1 = Dropout(dr)(dense1, training=True)
    dense2 = Dense(l2_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(dropout1)
    dropout2 = Dropout(dr)(dense2, training=True)
    dense3 = Dense(l3_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(dropout2)
    dropout3 = Dropout(dr)(dense3, training=True)
    dense4 = Dense(l3_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(dropout3)
    dropout4 = Dropout(dr)(dense4, training=True)
    output_layer = Dense(output_size, activation="tanh")(dropout4)
    model_nn = Model(input=[input_layer], output=[output_layer])
    model_nn.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model_nn


class NNPolicy:
    def __init__(self, input_size, output_size):
        self.nn = create_model(input_size, output_size)
        self.state_mean, self.state_std = None, None
        self.action_mean, self.action_std = None, None

    def normalise(self, states, actions):
        return (states-self.state_mean)/self.state_std, (actions-self.action_mean)/self.action_std

    def train(self, states, actions, EPOCHS):
        self.state_mean, self.state_std = np.mean(states, axis = 0), np.std(states, axis = 0)
        self.action_mean, self.action_std = np.mean(actions, axis = 0), np.std(actions, axis = 0)
        self.state_std[self.state_std == 0] = 1
        self.action_std[self.action_std == 0] = 1
        x, y = self.normalise(states, actions)
        self.nn.fit(x=x, y=y, shuffle=True, epochs=EPOCHS)
        return

    def predict(self, state):
        normalised_state = (state-self.state_mean)/self.state_std
        normalised_prediction = self.nn.predict(x=state)
        return (normalised_prediction*self.action_std) + self.action_mean

    def save(self):
        pickle.dump(self, open("model1", "wb"))


def main(EPOCHS):
    state_action_buffer = pickle.load(open(f"policy_data/state_action_buffer", "rb"))
    #delta_state_buffer = pickle.load(open(f"policy_data/delta_state_buffer", "rb"))
    states = state_action_buffer[:, :39]
    actions = state_action_buffer[:, 39:]
    input_size = states.shape[1]
    output_size = actions.shape[1]
    print(states.shape)
    print(actions.shape)
    nn_policy = NNPolicy(input_size, output_size)
    nn_policy.train(states, actions, EPOCHS)
    nn_policy.save()
    return


def main2():
    state_action_buffer = pickle.load(open(f"policy_data/state_action_buffer", "rb"))
    delta_state_buffer = pickle.load(open(f"policy_data/delta_state_buffer", "rb"))
    model = load_model("model1")
    states = state_action_buffer[:, :39]
    actions = state_action_buffer[:, 39:]
    #print("Test")
    print(model.evaluate(x=states[180000:], y=actions[180000:], verbose=False))
    print("Training")
    print(model.evaluate(x=states[:180000], y=actions[:180000], verbose=False))
    return


if __name__ == "__main__":
    main(1)


    #main2()

