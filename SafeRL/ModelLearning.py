import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.layers import Dropout, Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pickle

def create_model(input_size, output_size):
    #lr = 0.01
    dr = 0.2
    l1_dims, l2_dims, l3_dims = 300, 300, 300
    input_layer = Input(shape=(input_size,))
    dense1 = Dense(l1_dims, activation="relu")(input_layer)
    dropout1 = Dropout(dr)(dense1, training=True)
    dense2 = Dense(l2_dims, activation="relu")(dropout1)
    dropout2 = Dropout(dr)(dense2, training=True)
    dense3 = Dense(l3_dims, activation="relu")(dropout2)
    dropout3 = Dropout(dr)(dense3, training=True)
    output_layer = Dense(output_size, activation="linear")(dropout3)
    model_nn = Model(input=[input_layer], output=[output_layer])
    model_nn.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model_nn

def main(EPOCHS):
    state_action_buffer = pickle.load(open(f"policy_data/state_action_buffer", "rb"))
    delta_state_buffer = pickle.load(open(f"policy_data/delta_state_buffer", "rb"))
    print(state_action_buffer.shape)
    states = state_action_buffer[:, :47]
    actions = state_action_buffer[:, 47:]
    print(actions.shape)
    input_size = states.shape[1]
    output_size = actions.shape[1]
    model = create_model(input_size=input_size, output_size=output_size)
    print("Prior prediction")
    print(model.predict(x=states[:1]))
    model.fit(x=states[:80000], y=actions[:80000], shuffle=True, epochs=EPOCHS)
    print("Posterior predictions")
    for i in range(5):
        print(model.predict(x=states[:1]))
    print("Actual")
    print(actions[:2])
    model.save("model1")
    return


if __name__ == "__main__":
    main(10)

