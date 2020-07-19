---
layout: post
comments: true
title: Reinforcement Learning I: Foundation
date: 2020-07-10
author: Baihua Xie
tags: foundation reinforcement-learning
mins: long
---

> This is the first post in a series of (quite possibly) long reads that seek to briefly cover some of the core topics in deep reinforcement learning. As a starting point, this post will introduce the set up of the RL problem, key concepts, some of the classic algorithms, and several case studies from recent literature that showcase the wide-ranging applications of deep reinforcmenet learning. 



### Set up of the RL problem



##### Notations

| Symbol                              | Meaning                                                      |
| ----------------------------------- | :----------------------------------------------------------- |
| $$s, s'\in \mathcal{S}$$            | the current **state** $$s$$ and the next state $$s'$$ in the set of all possible states, i.e., the state space $$\mathcal{S}$$ |
| $$ws, as$$                          | world-state $$ws$$ and agent-state $$as$$                    |
| $$a\in \mathcal{A}$$                | an **action** $$a$$ in the set of all possible actions, i.e., the action space $$\mathcal{A}$$ |
| $$o_t$$                             | observation $$o$$ at time step $$t$$                         |
| $$r=R(s,a)$$                        | the **reward** $$r$$, given as a distribution $$R$$ over the state $$s$$ and the action $$a$$ |
| $$s_t, a_t, r_t$$                   | the state, action and reward at time step $$t$$              |
| $$h_t=\{a_1, o_1, ..., a_t, o_t\}$$ | history                                                      |
| $$P(s'|s,a)$$                       | **transition dynamics** $$P$$: the probability of the next state being $$s'$$ given the current state $$s$$ and action $$a$$ |
| $$\gamma$$                          | **discount factor**                                          |
| $$\tau$$                            | horizon: the last time step in the process; could be $$+\infty$$ |
| $$G_t$$                             | **return**: the sum of discounted future rewards from time step $$t$$ to the horizon; or $$G_t=\sum\limits_{i=0}^{+\infty}\gamma^ir_{t+i}$$ |
| $$\pi(a|s)$$                        | stochastic **policy function**: probability of action $$a$$ given the current state $$s$$ |
| $$V(s)$$                            | **state-value function**: the **expected** return over all possible future sequences starting from state $$s$$  <br />$$V(s)=\mathbb{E}[G_t|s_t=s]$$ |
| $$V^\pi(s)$$                        | state-value function following policy $$\pi$$ <br />$$V^\pi(s)=\mathbb{E}_{a\in\pi}[G_t|s_t=s]$$ |
| $$Q(s,a)$$                          | **action-value function**: the **expected** return over all possible future sequences starting from state $$s$$ and action $$a$$<br />$$Q(s,a)=\mathbb{E}[G_t|s_t=s,a_t=a]$$ |
| $$Q^\pi(s,a)$$                      | action-value function following policy $$\pi$$<br />$$Q^\pi(s,a)=\mathbb{E}_{a\in\pi}[G_t|s_t=s,a_t=a]$$ |
|                                     |                                                              |
|                                     |                                                              |



##### Sequential Decision Process



###### World



###### Agent



###### State



###### Observability



###### History



###### Markov Property



##### Model



###### Reward



###### Transition



##### Markov Decision Process



##### Value Function



##### Policy Function



##### Optimal Policy



##### Bellman Equation



##### Tabular Representation vs. Function Approximation



##### Bias vs. Variance



##### Exploration vs. Exploitation



##### Convergence



##### Terminology



### Classic Approaches to RL



##### Dynamic Programming



###### Value Iteration



###### Generalized Policy Iteration



##### Monte-Carlo Methods



##### Temporal-Difference (TD) Methods



###### SARSA: on-policy TD control



###### Q-learning: off-policy TD control



###### TD($$\lambda$$)



##### Comparison of DP, MC, TD Methods



### Model-free Approaches



##### Policy Gradients



###### Policy Gradient Theorem



###### REINFORCE



###### Actor-Critic



###### TRPO



###### A3C



###### PPO



##### Deep Q-learning



###### Case Study 1: DQN



###### Double DQN



###### Rainbow DQN



### Model-based Approaches



##### Model Is Given



###### Case Study 2: AlphaGo Zero



##### Model Is Learned



###### MBMF



###### MVE



### Evolution Strategies





### Exploration Strategies



### Known Problems in RL



### Case Study



###### Case Study 3: Neural Architecture Search



###### Case Study 4: Physical Design With Deep Reinforcement Learning





### References











