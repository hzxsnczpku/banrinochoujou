# 万里の長城
これは　万里の長城　です〜

Here is a Pytorch implementation of the Reinforcement Learning Algorithms.

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

### Distributions
* Discrete
* DiagGaussian
* DiagBeta
* Gaussian
* Dirichlet

## TO BE IMPLEMENTED
### Algorithms
*  DDPG
*  Q Learning with a Duel Structure
*  CEM
*  acktr
*  Distributional Q Learning
*  Feudal Network

### Modules
*  ICM

## How to Play
For example, run the following code to train a TRPO Agent under the MuJoCo HalfCheetah-v1 environment:
```
python main.py --env HalfCheetah-v1 --agent TRPO_Agent --use_mujoco_setting True --save_every 300
```

To get a more detailed overview of the parameters, run the following code:
```
python main.py -h
```

## Experiment Results
Below are some experimental results achieved by my baselines:

### MuJoCo Benchmark
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/HalfCheetah.png" width='400px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/Swimmer.png" width='400px'>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/banrinochoujou/master/images/InvertedPendulum.png" width='400px'>
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
