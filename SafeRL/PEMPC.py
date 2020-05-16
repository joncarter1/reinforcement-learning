from handful_of_trials.dmbrl.modeling.layers import FC
from handful_of_trials.dmbrl.modeling.models import BNN
from dotmap import DotMap
import tensorflow as tf

pe_params = DotMap(name="model", num_networks=5, sess=None, load_model=None, model_dir="PETS")

pe_model = BNN(pe_params)

state_dims = 10
action_dims = 2

pe_model.add(FC(200, input_dim=state_dims+action_dims, activation="swish", weight_decay=0.000025))
pe_model.add(FC(200, activation="swish", weight_decay=0.00005))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(200, activation="swish", weight_decay=0.000075))
pe_model.add(FC(state_dims, weight_decay=0.0001))
pe_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})