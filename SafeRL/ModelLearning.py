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
    dr = 0.1
    l2_penalty = 0.001
    l1_dims, l2_dims, l3_dims = 300, 300, 300
    input_layer = Input(shape=(input_size,))
    dense1 = Dense(l1_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(input_layer)
    dropout1 = Dropout(dr)(dense1, training=True)
    dense2 = Dense(l2_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(dropout1)
    dropout2 = Dropout(dr)(dense2, training=True)
    dense3 = Dense(l3_dims, kernel_regularizer=l2(l2_penalty), bias_regularizer=l2(l2_penalty), activation="relu")(dropout2)
    dropout3 = Dropout(dr)(dense3, training=True)
    output_layer = Dense(output_size, activation="tanh")(dropout3)
    model_nn = Model(input=[input_layer], output=[output_layer])
    model_nn.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model_nn

def main(EPOCHS):
    state_action_buffer = pickle.load(open(f"policy_data/state_action_buffer", "rb"))
    delta_state_buffer = pickle.load(open(f"policy_data/delta_state_buffer", "rb"))
    print(state_action_buffer.shape)

    states = state_action_buffer[:, :47]
    actions = state_action_buffer[:, 47:]
    input_size = states.shape[1]
    output_size = actions.shape[1]
    model = create_model(input_size=input_size, output_size=output_size)
    model.fit(x=states[:180000], y=actions[:180000], shuffle=True, epochs=EPOCHS)
    model.save("model1")
    return


def main2():
    state_action_buffer = pickle.load(open(f"policy_data/state_action_buffer", "rb"))
    delta_state_buffer = pickle.load(open(f"policy_data/delta_state_buffer", "rb"))
    model = load_model("model1")
    states = state_action_buffer[:, :47]
    actions = state_action_buffer[:, 47:]
    print("Test")
    print(model.evaluate(x=states[180000:], y=actions[180000:], verbose=False))
    print("Training")
    print(model.evaluate(x=states[:180000], y=actions[:180000], verbose=False))
    return


if __name__ == "__main__":
    #main(5)


    main2()

