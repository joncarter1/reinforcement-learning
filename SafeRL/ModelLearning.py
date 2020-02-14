import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.layers import Dropout, Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import safety_gym
import gym
import numpy as np
from safety_gym.envs.engine import Engine
from SafeRL.Controller import human_policy
import pickle

def create_model(input_size, output_size):
    #lr = 0.01
    dr = 0.001
    l1_dims, l2_dims = 400, 400
    input_layer = Input(shape=(input_size,))
    dense1 = Dense(l1_dims, activation="relu")(input_layer)
    dropout1 = Dropout(dr)(dense1, training=True)
    dense2 = Dense(l2_dims, activation="relu")(dense1)
    dropout2 = Dropout(dr)(dense2, training=True)
    output_layer = Dense(output_size, activation="linear")(dropout2)
    model_nn = Model(input=[input_layer], output=[output_layer])
    model_nn.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model_nn

def main(EPOCHS):
    state_action_buffer = np.array(pickle.load(open(f"policy_data/state_action_buffer", "rb")))
    delta_state_buffer = np.array(pickle.load(open(f"policy_data/delta_state_buffer", "rb")))
    input_size = state_action_buffer.shape[1]
    output_size = delta_state_buffer.shape[1]
    model = create_model(input_size=input_size, output_size=output_size)
    print("Prior prediction")
    print(model.predict(x=state_action_buffer[:1]))
    model.fit(x=state_action_buffer[:6000], y=delta_state_buffer[:6000], shuffle=True, epochs=EPOCHS)
    print("Posterior prediction")
    print(model.predict(x=state_action_buffer[:1]))
    print("Delta")
    print(state_action_buffer[1]-state_action_buffer[0])
    model.save("model1")
    return




if __name__ == "__main__":
    main(100)

