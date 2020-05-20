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

no_eps = 10

pe_params = DotMap(name="policy_model_{}".format(no_eps), num_networks=5, sess=None, load_model=None, model_dir="bnn_models")
pe_model = BNN(pe_params)

state_dims = 10
action_dims = 2

pe_model.add(FC(200, input_dim=state_dims, activation="swish", weight_decay=0.000025))
pe_model.add(FC(200, activation="swish", weight_decay=0.00005))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(output_dim=action_dims, weight_decay=0.0001))
pe_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})


if __name__ == "__main__":
    policy_data = np.array(pickle.load(open("SafeRL/policy_data/{}eps".format(no_eps), "rb")))
    state_vector = policy_data[:, :state_dims]
    action_vector = policy_data[:, state_dims:state_dims+action_dims]
    input_mean = np.mean(state_vector, axis=0)
    input_std = np.std(state_vector, axis=0)
    output_mean = np.mean(action_vector, axis=0)
    output_std = np.std(action_vector, axis=0)
    state_vector = (state_vector-input_mean)/input_std
    action_vector = (action_vector-output_mean)/output_std
    pe_model.train(inputs=state_vector, targets=action_vector, epochs=50, holdout_ratio=0.1)
    pe_model.save()
