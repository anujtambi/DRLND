
[image1]: https://github.com/anujtambi/DRLND/blob/master/DRLND/P1_Navigation/images/training.png "Training"

# Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation - report

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
