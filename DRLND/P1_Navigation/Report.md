
[image1]: https://github.com/anujtambi/DRLND/blob/master/DRLND/P1_Navigation/images/training.png "Training"

# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation - report

## Learning algorithm

In order to have room for improvements, the agent was trained using a fixed Q-target with experience replay. Having the parameter update from Temporal Difference-learning,
$$
\Delta w = \alpha \Big(R + \gamma \max_a \hat{q}(S', a, w) - \hat{q}(S, A, w) \Big) \nabla_w \hat{q}(S, A, w)
$$
the TD target is supposed to be a replacement for the true value of  $q_\pi(S, A)$ . But it actually is dependent on $w$ and so the derivative of the objective function is incorrect.

To fix this, the TD target becomes
$$
R + \gamma \max_a \hat{q}(S', a, w^-)
$$
where $w^-$ is a copy of $w$ that won't change during the learning step. Additionally, experience replay is a buffer mechanism to provide a past experience sampler for the agent to learn.



**Model architecture**

The model architecture is a succession of 3 fully connected layers (the hidden layer having 64 input and output features), with ReLU activations. Optionally, batch normalization and dropout layers can be added after penultimate layers. This was added to prevent potential overfitting, which did not turn out to be a problem to reach the target score.



### Solution
To execute this project a DQN algorithm was implemented.

### Performance
The agent achieved average score of more than 15 over 100 (as described below) consecutive episodes after 612 episodes of training.

Episode 100	Average Score: 1.16
Episode 200	Average Score: 4.58
Episode 300	Average Score: 8.47
Episode 400	Average Score: 11.10
Episode 500	Average Score: 12.31
Episode 600	Average Score: 14.39
Episode 700	Average Score: 14.49
Episode 712	Average Score: 15.02
Environment solved in 612 episodes!	Average Score: 15.02

![Training][image1]

### Following hyperparameters were used for the DQN Agent-
BUFFER_SIZE = int(1e5)  # replay buffer size

BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR = 5e-4               # learning rate 

UPDATE_EVERY = 4        # how often to update the network

### Ideas for Future Work
* Implement Prioritize Experience Replay
* Implement Dueling Architecture
* Implement n-step bootstrapping and/or other techniques described in Rainbow
