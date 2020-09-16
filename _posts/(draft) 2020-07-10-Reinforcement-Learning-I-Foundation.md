---
layout: post
comments: true
title: Foundations of Deep Reinforcement Learning
date: 2020-09-13
author: Baihua Xie
tags: foundation drl
mins: long
---

> This is the first post in a series of (quite possibly) long reads that seek to briefly cover some of the core topics in deep reinforcement learning. As a starting point, this post will introduce the set up of the RL problem, key concepts, some of the classic algorithms, and several case studies from recent literature that showcase the wide-ranging applications of deep reinforcement learning. 

This post would adopt the notation conventions as set forth by Richard Sutton in his seminal textbook<sup>1</sup>, as follows:

| Symbol                              | Meaning                                                      |
| ----------------------------------- | :----------------------------------------------------------- |
| $$s, s'\in \mathcal{S}$$            | the current **state** $$s$$ and the next state $$s'$$ in the set of all possible states, i.e., the state space $$\mathcal{S}$$ |
| $$a\in \mathcal{A}$$                | an **action** $$a$$ in the set of all possible actions, i.e., the action space $$\mathcal{A}$$ |
| $$o_t$$                             | observation $$o$$ at time step $$t$$                         |
| $$r=R(s,a)$$                        | the **reward** $$r$$, given as a distribution $$R$$ over the state $$s$$ and the action $$a$$ |
| $$s_t, a_t, r_t$$                   | the state, action and reward at time step $$t$$              |
| $$h_t=\{a_1, o_1, ..., a_t, o_t\}$$ | history                                                      |
| $$P(s'|s,a)$$                       | **transition dynamics** $$P$$: the probability of the next state being $$s'$$ given the current state $$s$$ and action $$a$$ |
| $$\gamma$$                          | **discount factor**                                          |
| $$\tau$$                            | horizon: the last time step in the process; could be $$+\infty$$ |
| $$G_t$$                             | **return**: the sum of discounted future rewards from time step $$t$$ to the horizon; or $$G_t=\sum\limits_{i=0}^{+\infty}\gamma^ir_{t+i}$$ |
| $$\pi(a|s)$$                        | stochastic **policy function**: probability of action $$a$$ given the current state $$s$$<br />function approximation: $$\pi_\theta(a|s)$$ |
| $$V(s)$$                            | **state-value function**: the **expected** return over all possible future sequences starting from state $$s$$  <br />$$V(s)=\mathbb{E}[G_t|s_t=s]$$ |
| $$V^\pi(s)$$                        | state-value function following policy $$\pi$$ <br />$$V^\pi(s)=\mathbb{E}_{a\in\pi}[G_t|s_t=s]$$ |
| $$Q(s,a)$$                          | **action-value function**: the **expected** return over all possible trajectories starting from state $$s$$ and action $$a$$<br />$$Q(s,a)=\mathbb{E}[G_t|s_t=s,a_t=a]$$ |
| $$Q^\pi(s,a)$$                      | action-value function following policy $$\pi$$: $$Q^\pi(s,a)=\mathbb{E}_{a\in\pi}[G_t|s_t=s,a_t=a]$$ |
| $$A^\pi(s,a)$$                      | **advantage function**: $$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$$   |
|                                     |                                                              |

##### Set up of the RL problem

Reinforcement learning deals with a system where an __agent__ interacts with its __environment__ in a __sequential process__. At any given __timestep__ $$t$$ in the process, the system is characterized by a sufficient information representation known as the __state__ $$s_t$$ . The agent is an active learner, which could be trained to perform a certain __action__ $$a_t$$ given the current state $$s_t$$. The mapping from states to actions is known as the __policy function__. An action subsequently induces a transition of the environment's state from the current state $$s_t$$ to the next state $$s_{t+1}$$ (or $$s'$$ for short), as well as a __reward__ signal $$r_{t+1}$$. The transition and reward are characterized by a __transition dynamics model__ (or "model" for short). After a sequence of interactions, a __trajectory__ could be recorded as $$\{s_0, a_0, r_1, s_2, a_2, r_2, ...\}$$.

Given a trajectory, any state $$s_t$$ in the trajectory is associated with a __return__, denoted by $$G_t$$, the sum of future rewards in the trajectory starting from state $$s_t$$. The return quantifies the "value" of being in the state $$s_t$$, if the future sequence of states, actions and rewards follow the trajectory. Notice, however, that the return of a state $$s_t$$ is not deterministic, as different trajectories with different returns could stem from the same state. In principle, it is desirable to know the "value" of a state $$s_t$$, known as the __(state) value function__, by quantifying the _expected_ return from state $$s_t$$ over all possible trajectories.

With the above definitions, we can now frame the RL problem as the following:

Given an environment that could interact with the agent and access to the trajectories following such interactions, the goal of training a RL agent is to learn a policy function such that by following it the agent interacts with the environment in a way that maximizes the value function, for any given state.

Now we are ready for a more formal treatment of the various elements in a RL algorithm:

__Observability__. Previously it is implicitly assumed that the state $$s_t$$ is the same for the agent and the environment. A further distinction could be drawn between the agent's state and the environment's state (the "world state"). This is because the true state of the environment may not always be fully observable to the agent. Strictly speaking, the agent is really receiving an observation $$o_t$$ through its interaction with the environment. Generally we can assume that the representation describing an agent's state is equivalent to that describing the observation, as this is really all that the agent can know about the environment (besides reward); typically this representation is simply referred to as the state $$s_t$$ without further distinctions. However, there are two types of environments, depending on whether or not the same is true for the world state:

* __Fully-observable__ environments are where the world state is also equivalent to the observation. In these environments, the concept "state" simply refers to a representation that describes the agent state or the world state, as they are equivalent. A typical example would be games like chess or Go, where everything on the board (the environment) is observable to either of the players (the agents). _Unless otherwise specified, the rest of the post would deal with fully-observable environments only_.
* __Partially-observable__ environments are where the world state is not fully represented by the observation. The precise definition of the concept "state" in these environments would require more careful treatments. A common way<sup>cite</sup> is to define the state $$s_t$$ as the __history__ $$h_t$$,which is a set of past actions, observations and rewards recorded before timestep $$t$$; this is to ensure the __Markov property__ for partially-observable environments. A typical example of such environments would be games like Poker, where the players do not have information on the other players' cards. 

__Markov Property__. The interaction between an agent and its environment is manifested in part by the fact that an agent's action could induce transition from current state $$s_t$$ to next state $$s_{t+1}$$ with an additional reward signal $$r_{t+1}$$. In general, the probability of transitioning to the state $$s_{t+1}$$ and reward $$r_{t+1}$$ is usually described by a conditional probability function of the action $$a_t$$ and history $$h_t$$, denoted as $$P(s_{t+1}, r_{t+1}|h_t, a_t)$$. An environment is said to satisfy the Markov property if $$P(s_{t+1}, r_{t+1}|h_t, a_t)=P(s_{t+1}, r_{t+1}|s_t, a_t)$$; that is, the environment's transition is fully characterized by the current state $$s_t$$. Aside from the obvious benefit that Markov property simplifies the problem setting considerably, this is an important property in reinforcement learning for several practical reasons also:

* many practical problems for reinforcement learning satisfies the Markov property
* an environment could always be made Markov by setting $$s_t = h_t$$
* some of the fundamental algorithms and theories in RL assume that the environment is Markov

__MDP__. The sequence of interactions between an agent and its environment forms what is knowns as a sequential process. A sequential process might be characterized by the nature of the interaction (what statistics are generated):

* The most basic sequential process is characterized by interactions involving simply a state space $$S$$ and the transition probability $$P$$, denoted by $$(S,P)$$. If the environment satisfies the Markov proper, this is also referred to as the Markov process.
* If a reward signal (from a reward space $$R$$) is given after each interaction, the process is knowns as a Markov reward process. It is common to associate with the rewards a discount factor $$\gamma<1$$, such that rewards expected further in the future would discounted more severely (more details in following sections). The Markov reward process could be characterized by a tuple of $$(S, R, P, \gamma)$$.
* A __Markov Decision Process (MDP)__ further involves actions (from action space $$A$$) selected by the agent and is characterized by $$(S,A,R,P,\gamma)$$. MDP's are usually the implicit underlying assumption that most of the common RL algorithms work with.

__Model__. The transitions in the environment is assumed to be governed by a transition dynamics model, denoted as $$p(s',r|s,a)$$. The subscript $$t$$ has been omitted for simplicity (here using the four-argument function; following Sutton). The model describes the conditional probabilities of transitioning into state $$s'$$ and reward $$r$$ given the current state $$s$$ and agent's action $$a$$. Although in theory the model is an essential part to an MDP, in practice it is often the case that the underlying dynamics model of an environment would be unknown to the algorithm. A RL algorithm could be categorized as: 

* _model-free_ if it does not require an explicit dynamics model. Instead, it relies solely on interactions with the environment and observing the states and rewards. Some of the most successful RL methods at the moment, including deep Q-learning<sup>cite</sup> and policy gradient<sup>cite</sup> methods, are model-free.
* _model-based_ if it requires a dynamics model. Model-based methods have been gaining more attention in recent literature. There would be a later section dedicated to some major model-based works.

__Policy__. The mapping from states to actions is known as the policy function, denoted by $$\pi(a|s)$$. Policy function describes the conditional probability of selecting action $$a$$ given the current state $$s$$. 

__Value__. 

The ultimate goal of a RL algorithm is to train the agent to learn an optimal policy function, $$\pi^*(a|s)$$, such that it selects a sequence of actions in a way that maximizes the __value function__

A key concept in the RL problem is the __value function__. A value function is a function of the agent's state, denotes as $$V(s)$$, which equals to the __expected__ summation of all future rewards from state $$s$$. 

A further distinction might be made for the state of the agent ('agent state') and the state of the environment ('world state'). 

### References











