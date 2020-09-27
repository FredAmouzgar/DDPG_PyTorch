# Deep Deterministic Policy Gradient (DDPG) implemeted by PyTorch for the Unity-based Reacher Envirnment

## Introduction
This repository is an implementation of the DDPG algorithm for the Reacher Environment developed by Unity3D and accessed through the UnityEnvironment library. It is an extension of the code sample provided by the Udacity Deep RL teaching crew (for more information visit their [website](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)). The environment is presented as a vector; thus, we did not use Convolutional Neural Networks (CNN) in the implementation.

This repository consists of these files:

*These files are saved under the "src" directory.*
1. <ins> model.py </ins>: This module provides the underlying neural network for our agent. When we train our agent, this neural network is going to be updated by backpropagation.
2. <ins>replay_buffer.py</ins>: This module implements the "memory" of our agent, also known as the Experience Replay.
3. <ins>agent.py</ins>: This is the body of our agent. It implements the way the agent acts using an actor-critic paradigm, and learn an optimal policy.
4. <ins>train.py</ins>: This module has the train function which takes the agent, the environment, number of training episodes and the required hyper-parameters and trains the agent accordingly.

To test the code, after cloning the project, open the `Reacher_Continuous_Control.ipynb` notebook. It has all the necessary steps to install and load the packages, and train and test the agent. It also automatically detects the operating system, and loads the corresponding environment. There is an already trained agent stored in `checkpoint-actor.pth` and `checkpoint-critic.pth`, by running the last part of the notebook, this can be directly tested.

## The Reacher Environment
The example uses a modified version of the Unity ML-Agents Reacher Example Environment. The environment includes In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The environment uses multiple unity agents to increase training time.

<img src="https://github.com/FredAmouzgar/DDPG_PyTorch/raw/master/images/Reacher.png" width="400" height="200">

### Multiagent Traning
The Reacher environment contains multiple unity agents to increase training time. The training agent collects observations and learns from the experiences of all of the unity agents simultaneously. The Reacher environment example employed here has 20 unity agents.

### State and Action Space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## A Smart Agent
Here is a reward plot acquired by the agent while learning. It surpasses +33 after around 120 episodes.
<img src="https://github.com/FredAmouzgar/DDPG_PyTorch/raw/master/images/DDPG_reward_plot.png" width="400" height="200">

Look at it go:

<img src="https://github.com/FredAmouzgar/DDPG_PyTorch/raw/master/images/Reacher.gif" width="400" height="200">