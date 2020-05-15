[//]: # (Image References)

[image1]: https://github.com/nitink12/DeepReinforcementLearningNanoDegree/blob/master/P3_Collab-Combat/images/environment_p3.gif "Trained Agent"

# Project 3: Collaboration and Competition

### Introduction

In this project, the goal is to teach two agents to move two tennis rackets so that they bounce a ball over a net between them for as long as possible. Each of the two agents has access to the environment's 8 dimensional state, which consists of the position and velocity of both the ball and racket. Each of these 8 dimensional states takes on continuous values for each racket agent. From this state, the agent learns which of four actions it should take. Each agent receives a reward of +0.1 if the agent his the ball over the net. If the agent lets the ball his the ground or bounces it out of bounds, it receives a rewar of -0.01. Each agent has two continuous actions available, one for moving towards or away from the net, and one for jumping. The agents have solved the environment when they average a score of +0.5 over 100 consecutive episodes (where the maximum is taken over the two agents).


![Trained Agent][image1]

### Getting Started


1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  
