DQN implementation architecture

The epsilon-greedy policy was used, with epsilon decayed to zero over the first 1000 episodes. 

A 2 layer neural network with 50 nodes per layer was used to approximate the Q-function, 
with soft-max activation between hidden layers and linear activation at the output since
it is fitting to a continous value.
