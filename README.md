# 万里の長城
これは　万里の長城　です〜

Here is a Pytorch implementation of the Reinforcement Learning Algorithms.

## News: Currently doing some mujoco experiments with the DDPG algorithm.
I am tuning the DDPG algorithm on the swimmer environment currently.

## News: My implementation of the Bayesian methods in Q-Learning
I recently have done some simple experiments on the Bayesian methods in Q-Learning. My main ideas are borrowed form the following three papers,

* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
* [Weight uncertainty in neural networks](https://arxiv.org/abs/1505.05424)
* [An empirical evaluation of thompson sampling](http://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf)

I try the variance inference approach and the dropout approach on the CartPole, Acrobot and nChain environment. The results of different algorithms are shown in the following figures and tables,

### nChain
| N | 20 | 30 | 50 | 80 | 100 |
|:-:|:--:|:--:|:--:|:--:|:---:|
|Bayesian TS|20.0|14.05|50.0|80.0|80.15|
|Bayesian Dropout|19.05|30.00|45.10|76.05|80.20|
|DQN no noise|14.30|6.00|15.50|4.00|60.00|
|DQN ε-greedy|9.00|9.05|10.35|24.25|40.25|

### Classical Control
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/CartPole" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Acrobot" width='280px'>
</div>

## Basic Agents & Modules
### Algorithms
* [Deep Q Learning](https://arxiv.org/abs/1312.5602)
* [Double Deep Q Learning](https://arxiv.org/abs/1509.06461)
* [Deep Q Learning with the Priorized Replay Memory](https://arxiv.org/abs/1511.05952)
* [Asynchronous Advantage Actor-Critic](https://arxiv.org/abs/1602.01783)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
  * clipped surrogate loss
  * adapted surrogate loss
* [Evolution Strategy](https://arxiv.org/abs/1703.03864)
* [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)

### Distributions
* Discrete
* DiagGaussian
* DiagBeta

## TO BE IMPLEMENTED
### Algorithms
*  Q Learning with a Duel Structure
*  CEM
*  acktr
*  Distributional Q Learning
*  Feudal Network

### Distributions
* Gaussian
* Dirichlet

### Modules
* VIME
* ICM

## How to Play
For example, run the following code to train a TRPO Agent under the MuJoCo HalfCheetah-v1 environment:
```
python main.py --env HalfCheetah-v1 --agent TRPO_Agent --use_mujoco_setting True --save_every 300
```

To get a more detailed overview of the parameters, run the following code:
```
python main.py -h
```

* I have change the structure of the code, so the above instructions no longer works, an alternative one will soon be given.

## Experiment Results
Below are some experimental results achieved by my baselines:

### MuJoCo Benchmark
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/HalfCheetah.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Swimmer.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/InvertedPendulum.png" width='280px'>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Reacher.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/InvertedDoublePendulum.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Hopper.png" width='280px'>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Walker2d.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Humanoid.png" width='280px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/HumanoidStandup.png" width='280px'>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Ant.png" width='280px'>
</div>

### Atari Benchmark
Under construction...

## Dependency
* tabulate 0.7.7
* scipy 0.19.0
* pytorch 0.2.0
* pyparsing 2.1.4
* openai gym

## References
Under Construction...
