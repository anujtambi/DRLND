## P3 - Tennis
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Learning Algorithm
The Learning algorithm used in this project is a modification of the implementation used in the [pendulum environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).

DDPG is a policy based learning algorithm in which the agent will learn from the nonprocessed observation spaces without knowing the environment. In contrast to a DQN which learns directly through a gradient method which estimates the weights of the optimal policy. 

DDPG employs an Actor-Critic model in which the Critic model learns the state-value function and uses this to determine how the Actor's policy model should change. The Actor learns from the continuous space without the needs for many data samples since it can rely on the critic to give it feedback on good and bad actions.

Each agent uses the same actor network to take an action, sampled from a shared replay buffer. However, stability could be a problem with this approach. You can end up with a model that learns well and crashes after a few episodes. To mitigate these, there are several techniques that can be employed like, Gradient Clipping, Soft Target Update, Twin local/ target networks, and Replay Buffer. Reply Buffer is most critical as it allows the DDPG agent to learn offline by gathering experiences collected from the environment agents and sample experiences from a large memory buffer across experiences.

### Model Architecture

The algorithm uses two deep neural networks (actor-critic) with the following struture:
- Actor    
    - Hidden: (input, 256)  - ReLU
    - BatchNormalization: (256)
    - Hidden: (256, 256)    - ReLU
    - Output: (256, 2)      - TanH   # action_size=2

- Critic
    - Hidden: (input, 256)              - ReLU
    - BatchNormalization: (256)
    - Hidden: (256 + action_size, 256)  - ReLU
    - Output: (256, 1)                  - Linear

## Hyperparameters
- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 8e-5        # learning rate of the actor 
- LR_CRITIC = 8e-5        # learning rate of the critic
- WEIGHT_DECAY = 0.00000   # L2 weight decay

## Results 

Episode 2414	Average Score: 1.0074000150337816	 Max Score: 2.60000003874301922642

Envionment solved after 2414 episodes

![Training](https://github.com/nitink12/DeepReinforcementLearningNanoDegree/blob/master/P3_Collab-Combat/images/training.png)

## future improvements
1. Attempt and solve the Football environment. I think this would be challenging as the environment not only collaborative between players in the same team but is also competitive between players of opposite teams.
2. Further Reduce learning time by more adjusting hyper parameters and tuning the network.
