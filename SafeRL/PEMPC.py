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
from SafeRL.ModelLearning import gaussian_nll
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"gaussian_nll": gaussian_nll})

pe_params = DotMap(name="bnn_model3", num_networks=5, sess=None, load_model=None, model_dir="bnn_models")

pe_model = BNN(pe_params)

state_dims = 10
action_dims = 2
combined_dims = state_dims+action_dims

pe_model.add(FC(200, input_dim=combined_dims, activation="swish", weight_decay=0.000025))
pe_model.add(FC(200, activation="swish", weight_decay=0.00005))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(output_dim=state_dims, weight_decay=0.0001))
pe_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
#pe_model.finalize(tf.train.RMSPropOptimizer, {"learning_rate": 0.001})


if __name__ == "__main__":
    ll = pickle.load(open("SafeRL/MPCModels/pempc_model", "rb"))
    model_data = ll.REPLAY_MEMORY
    state_action_vector = np.array(ll.REPLAY_MEMORY)[:, :ll.combined_dims]
    delta_state_vector = np.array(ll.REPLAY_MEMORY)[:, -ll.state_dims:]
    input_mean = np.mean(state_action_vector, axis=0)
    input_std = np.std(state_action_vector, axis=0)
    output_mean = np.mean(delta_state_vector, axis=0)
    output_std = np.std(delta_state_vector, axis=0)
    state_action_vector = (state_action_vector-input_mean)/input_std
    delta_state_vector = (delta_state_vector-output_mean)/output_std
    pe_model.train(inputs=state_action_vector, targets=delta_state_vector, epochs=25, holdout_ratio=0.1)
    pe_model.save()
