
[image1]: https://github.com/anujtambi/DRLND/blob/master/DRLND/P1_Navigation/images/training.png "Training"

# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation - report

## Reinforcement Learning

In a Reinforcement Learning problem an *agent* observes the *state* _s(t)_ of an *environment* in a given moment of time, and chooses an *action* _a(t)_. When executed the action changes the environment moving it to a new state _s(t+1)_, and the action receives a *reward* _r(t+1)_. The objective of the agent is maximize the reward received after multiple interactions with the environment. In order to achieve this goal the agent must *explore* the environment in  order to find out the best actions given the state, and also to *exploit* the acquired knowledge to collect the rewards.  

## Q-Learning and Q-Networks 

A *policy* is a function that gives the probability of choosing an action _a_ when observing a state _s_. The *optimal policy* is the one that maximizes the estimated reward obtained when following the policy. 

In Q-learning the agent tries to find the optimal action-value function Q, which maps a (state,action) pair to the estimated reward obtained when following the optimal policy. If the environment state is discrete, the action-value function Q can be represented as a table.

If the environment state is continuous, we need to approximate the action-value function Q. This can be done using a neural network as a non-linear function approximator, adjusting its weights according to the observed rewards.

### DQN

In the standard DQN the Q target value is updated based on the difference between the previous value provided by the network and the observed reward.


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
