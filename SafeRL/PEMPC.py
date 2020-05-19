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
pe_params = DotMap(name="bnn_model", num_networks=5, sess=None, load_model=None, model_dir="bnn_models")

pe_model = BNN(pe_params)

state_dims = 10
action_dims = 2
combined_dims = state_dims+action_dims
input_dims, output_dims = 1, 1

pe_model.add(FC(200, input_dim=combined_dims, activation="swish", weight_decay=0.000025))
pe_model.add(FC(200, activation="swish", weight_decay=0.00005))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
#pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(output_dim=state_dims, weight_decay=0.0001))
pe_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
#pe_model.finalize(tf.train.RMSPropOptimizer, {"learning_rate": 0.001})


if __name__ == "__main__":
    loaded_learner = pickle.load(open("SafeRL/MPCModels/mpcmodel1", "rb"))
    model_data = loaded_learner.REPLAY_MEMORY
    state_action_vector = np.array(model_data)[:, :combined_dims]
    delta_state_vector = np.array(model_data)[:, -state_dims:]
    print(state_action_vector.shape)
    print(delta_state_vector.shape)
    pe_model.train(inputs=state_action_vector, targets=delta_state_vector, epochs=1, holdout_ratio=0.1)
    pe_model.save()