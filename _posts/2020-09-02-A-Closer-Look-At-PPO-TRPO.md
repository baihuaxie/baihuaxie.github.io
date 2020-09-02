---
layout: post
comments: true
title: A Closer Look At PPO and TRPO
date: 2020-09-02
author: Baihua Xie
tags: deep-policy-gradients deep-rl
mins: 15-min
---

> This post briefly introduces the popular deep reinforcement learning algorithms, TRPO and PPO, from the theoretical grounds leading up to the development of the algorithms, to the various approximations necessary for the agents to work practically. As they became more mainstream, it is natural to raise a question central to all of these deep RL methods: to what extend does the algorithm in practice reflect the theoretical principles leading to its development? Recently, there have been several interesting works that explored in depth the empirical behaviors of the algorithms. It is suggested that the seemingly sound theoretical justifications for the algorithms often fail to manifest in practice, and consequently there are still a lot more to be understood as to why these algorithms perform well on certain benchmarks. 

The notations are summarized in the following table:

| Symbol                                                  | Meaning                                                      |
| ------------------------------------------------------- | :----------------------------------------------------------- |
| $$s_t, a_t, r_t$$                                       | the **state** $$s$$, **action** $$a$$ and **reward** $$r$$ at current timestep $$t$$ |
| $$\gamma$$                                              | **discount factor**                                          |
| $$\pi_\theta(a_t \vert s_t)$$                           | the policy network parameterized by $$\theta$$ that approximates the stochastic **policy function** |
| $$\hat{V}_\phi(s_t)$$                                   | the value network parameterized by $$\phi$$ that approximates the **state-value function** |
| $$\hat{Q}(s_t,a_t)$$                                    | the estimated **action-value function**                      |
| $$\hat{A}(s_t,a_t)$$                                    | the estimated **advantage function**                         |
| $$\hat{g}$$                                             | the estimated **policy gradient**                            |
| $$\mathbb{E}_{s,a\sim\pi}$$                             | the expectation over the possible trajectories of state-action pairs following policy $$\pi$$ |
| $$\hat{\mathbb{E}}_t=\frac{1}{T}\sum\limits_{t=1}^{T}$$ | approximate the true expectation by the average over a sample trajectory of length $$T$$ |

The basic RL problem setting is assumed in the discussion. The agent has access to a Markov Decision Process (MDP). At each iteration of the training process, the agent interacts with the environment and samples a trajectory of $$\{s_t, a_t, r_t\}$$ following current policy $$\pi_{\theta_{old}}$$ for $$T$$ timesteps. The samples are subsequently used to form estimates of the various quantities needed to optimize the parameters, such as advantage function and policy gradient. To encourage exploration, one could use asynchronous setting<sup>1</sup> or off-policy setting<sup>2</sup> through importance sampling (for the latter, typically with experience replay<sup>3</sup> to improve stability), though neither choice would alter the core algorithms significantly. Hence for the simplicity in discussion, this post would focus only on the on-policy version of the algorithms.

Policy gradient algorithms are a leading class of model-free reinforcement learning methods. Compared to deep Q-learning<sup>3</sup>, another class of very successful value-based methods, policy gradients are considered to be more soundly grounded theoretically. The theories can be traced back to the ground-breaking work of Richard Sutton et al on the policy gradient theorem<sup>4</sup>, which provided a theoretical formula for the gradients w.r.t. parameters of the policy network. Its generalized form is as follows<sup>5</sup>:

$$
\hat{g}=\mathbb{E}_{s,a\sim\pi_\theta}[\sum\limits_{t=0}^\infty\Psi_t\nabla_\theta\log\pi_\theta(a_t \vert s_t)]
$$

where $$\Psi_t$$ typically takes the form of a Q-value function, an advantage function, a TD-residual of value functions, or simply the sum of (discounted) rewards of the sampled trajectory, among other choices. Meanwhile, it is almost imperative in practice to subtract from $$\Psi_t$$ a baseline $$b_t$$, which is a function of the state $$s_t$$ at timestep $$t$$ (a natural choice would be the value function), in order to reduce the high variance of this gradient estimator. Evidently, this necessitates a value network to estimate the value function, in addition to a policy network in deep policy gradient methods. In practice, however, it is common to share the majority of the parameters for the two networks.

The policy gradient provides the direction to update the policy network parameters and is used by the update rule $$\theta\gets\theta+\alpha\Delta\theta$$. Although the size of the update is not theoretically specified by the policy gradient, naively, one could simply use $$\Delta\theta\gets\hat{g}$$ and leave the learning rate or step size $$\alpha$$ as a hyperparameter, as is the standard procedure in deep learning framework. Classical policy gradient algorithms such as REINFORCE adopted this formulation. However, in the context of policy gradient methods, the algorithms could benefit from a more careful consideration for the step size. 

The analysis starts with the implicit assumption behind the policy gradient theorem, in that it is the solution to a constrained optimization problem of the following format:

$$
\DeclareMathOperator*{\argmax}{argmax}\\
g\gets\argmax\limits_g J(\theta+g)\;\text{ subject to a constraint}\;||g||\le\epsilon
$$

where the objective function $$J(\theta)=V_{\pi_\theta}(s_t)$$ is the "true" state-value function following current policy $$\pi_\theta$$. The constraint applies a bound on the L2-norm (or Euclidian distance) of the parameter gradient $$g$$, which sets a maximum step size $$\epsilon$$. 

However, it is difficult to select a proper bound on the step size in practice.  If the step size is too small, the learning becomes very slow; if too large, the update may not be optimal or even positive. This difficulty arises with the fact that in the naive formulation of the constrained optimization problem, the constraint is bounded in the parameter space. What the algorithm really needed, though, is a constraint bounded in the policy function space directly. In this manner, the algorithm could have better control of the resulting updated policy function after each iteration; ideally, we want to ensure that the algorithm could take the largest possible step size (hence fastest learning speed) that satisfies the constraint on the resulting policy.

This train of thought led to the development of the natural policy gradient<sup>6</sup> methods, which is a solution to a different constrained optimization problem:

$$
\DeclareMathOperator*{\argmax}{argmax}\\
g\gets\argmax\limits_g J(\theta+g)\;\text{ subject to a constraint}\;D_{KL}[\pi_{\theta_{old}} \vert \pi_\theta]\le\delta
$$

where the objective function is the same as before, and the constraint is now a KL divergence on the two policy functions, as parameterized by the parameters before and after the current update, respectively. The KL divergence is a measure of distance between two probability distributions $$P$$ and $$Q$$ such that $$D_{KL}[P \vert Q]=\int\limits_{-\infty}^{+\infty}P(x)\log(\frac{P(x)}{Q(x)})dx$$. 

How to solve this problem? A general methodology for constrained optimization problems with inequalities is the so called Karush-Kuhn-Tucker (KKT) method, which is a generalization from the classic Lagrangian Multiplier method. As a result, if taken a quadratic approximation, this KL divergence could be related to the Fisher Information Matrix (FIM) by:

$$
\mathbb{E}_{s,a\sim\pi_{\theta_{old}}}[\nabla_\theta^2 D_{KL}(\pi_{\theta_{old}}(\cdot \vert s) \vert \pi_\theta(\cdot \vert s))]\vert_{\theta=\theta_{old}}=\mathbb{E}_{s,a\sim\pi_{\theta_{old}}}[\nabla_\theta\log\pi_\theta(\cdot \vert s)\nabla_\theta\log\pi_\theta(\cdot \vert s)^T]\vert_{\theta=\theta_{old}}=F(\theta_{old})
$$

In other words, the Hessian matrix of the KL divergence is equal to the covariate matrix of the policy gradient, which in turn is the definition of the FIM. Any algorithm based on the natural policy gradient is thus second-order, as it is required to compute the FIM, which can become prohibitively expensive to do so for large state spaces. In practice, it is usually preferable to compute the Hessian rather than the covariate thanks to the conjugate gradient algorithm<sup>7</sup>, which allows using the FIM in subsequent gradient estimation without actually computing the entire second-order matrix.

Following the solution, the natural policy gradient and the corresponding bound on the step size are given by the theorem as:

$$
\begin{split}
g_N&\gets F^{-1}\hat{g}\;\text{where }\hat{g}\text{ is vanilla policy gradient}\\
\theta&\gets\theta+\alpha g_N \;\text{where }\alpha=\sqrt{\frac{2\delta}{g_N^TF^{-1}g_N}}
\end{split}
$$

Either the vanilla policy gradient theorem or the natural gradient theorem forms the skeleton of a deep policy gradient algorithm. But for the algorithm to work in practice, more auxiliary accommodations needed to be made. Consequently, policy gradient algorithms could typically be distinguished by the following 4 key characteristics:

**The objective function.** At first glance it seemed trivial to always use the "true" value function as the objective for optimization, as is the case for the previous theorems. But a "true" value function can be very difficult to optimize, particularly for large state spaces. Modern deep policy gradient methods instead opted for optimizing a so called "surrogate" objective function. A surrogate objective could be expressed in a simpler and explicit form, thus simplifying the training process, while simultaneously it is guaranteed that an improvement in the surrogate objective always corresponds an improvement in the "true" objective, albeit within a certain boundary in the parameter space (the trust region). Or so the theory promises. In practical algorithms, however, it is seldom explicitly proven that the chosen surrogate objective guarantees improvements in the "true" value function; this is more of an implicit assumption rather than an explicit proof. 

* The TRPO surrogate objective<sup>7</sup>:
  
  $$
  L(\theta)^{TRPO}=\mathbb{E}_{s,a\sim\pi_{\theta_{old}}}[\frac{\pi_\theta(a \vert s)}{\pi_{\theta_{old}}(a \vert s)}\hat{Q}(s,a)]
  $$
  
  note that the Q-value function (as in original paper) could be replaced by the advantage function; this is the same surrogate as CPI<sup>11</sup>

* The PPO-clipped surrogate objective<sup>8</sup>:
  
  $$
  L(\theta)^{PPO-CLIP}=L^{CLP}(\theta)-c_1L^{VF}(\theta)+c_2S(\theta)
  $$
  
  where $$L^{CLP}(\theta)$$ is a clipped version of the TRPO surrogate, $$L^{VF}(\theta)$$ is the MSE loss for value network, and $$S(\theta)$$ is the mean entropy of the policy.

* The PPO-penalty surrogate objective<sup>8,9</sup>:

$$
L(\theta)^{PPO-PEN}=\mathbb{E}_{s,a\sim\pi_{\theta_{old}}}[\frac{\pi_\theta(a_t \vert s_t)}{\pi_{\theta_{old}}(a_t \vert s_t)}\hat{A}_t]-\lambda D_{KL}[\pi_{\theta_{old}} \vert \pi_{\theta}]
$$

**The gradient estimator.** For methods assuming the "true" (and implicit) value function as the objective, it is convenient to estimate policy gradient by either the vanilla policy gradient theorem or the more sophisticated natural policy gradient. Even with an explicit surrogate objective function, sometimes it is still difficult to express the policy gradient w.r.t. $$\theta$$ by simply taking $$\nabla_\theta L(\theta)$$, as an extra constraint (if exists) can complicate the solution. For instance, TRPO solves a KL-divergence constrained optimization problem, thus it still requires help from the natural policy gradient theorem to form its gradient estimator. On the other hand, by using a different surrogate function, PPO simultaneously removed the constraint altogether, thus for its unconstrained optimization problem, the gradient could be easily obtained directly from the surrogate. 

* the TRPO policy gradient:

$$
\Delta\theta=\alpha\hat{F}^{-1}\hat{g}
$$

* the PPO policy gradient:

$$
\Delta\theta=\nabla_\theta L(\theta)^{PPO}
$$

**The value estimator.** Both the choice of surrogate objective and the gradient estimator usually require estimates to some value function (state-value, action-value, or advantage value function). Since the surrogate objective should approach the "true" objective, which is a state value function, it is no surprise that most papers chose surrogate functions incorporating estimates to the value function in certain manners. As for the gradient estimator, as mentioned before, it is imperative to deduct a value function as baseline from the estimator to reduce variance in practice.

Value estimation is usually done by a value network, in addition to a policy network. The two networks can and often do share parameters. There are various ways for estimating the value function, the Q-value function and the advantage function, and they may differ yet again in off-policy settings for low-variance considerations. TRPO simply uses a Monte-Carlo estimate for its Q-value approximator, which is high-variance and works only on-policy. PPO and later works prefer to instead approximate the advantage function using the generalized advantage estimation<sup>5</sup>, where:

$$
\hat{A}_t^{GAE(\gamma,\lambda)}=\sum\limits_{l=0}^\infty(\gamma\lambda)^l\delta_{t+l}^V\\
$$

where $$\delta_t^V=r_t+\gamma\hat{V}_\phi(s_{t+1})-\hat{V}_\phi(s_t)$$ is the TD-residual between two successive states. This estimator has been widely adopted mainly because it is flexible; the hyperparameters $$\lambda,\gamma$$ facilitate smooth bias-variance tuning for the estimation. In theory, as the authors concluded in the GAE paper, it also only introduces a small (and diminishing w.r.t. longer trajectory lengths) bias into the gradient estimator compared to more traditional methods. 

When adapted to off-policy setting, one faces the challenge of higher variance. A commonly used off-policy estimator for the Q-value function is the Retrace algorithm by Munos et al (2016). The ACER paper<sup>2</sup> , which focuses on off-policy adaptation of policy gradient methods in actor-critic style, further borrowed an idea called stochastic dueling networks (SDNs) from Wang et al (2015) as an off-policy method to estimate the Q-value and the state-value function simultaneously in the continuous domain.

**The optimizer.** The optimizer describes the update rule for the trained parameters; in its simplest form, $$\theta\gets\theta+\alpha\Delta\theta$$. Although this seems very simple, for anyone coming to the deep RL world from a deep learning background, the (seemingly excessive) considerations for the optimizer is perhaps the most confusing aspect of a sophisticated algorithm. Deep learning people have familiarized themselves with out-of-the-box optimizers such as SGD, Adam, RMSprop, to name a few. The only aspects about optimization that people actually need to care about during training are the choice of batch size and the schedule of the learning rate, which incidentally also follow a straightforward inverse-linear relationship<sup>10</sup>.

The primary consideration that distinguishes deep reinforcement learning from deep learning in the context of optimization is the step size or the learning rate. Any gradient-based optimization method needs to consider two quantities: the direction of the update and the step of update. In deep learning, the direction is the gradient direction, which is also in theory the direction with steepest descent. The learning rate is implicitly assumed to be a hyperparameter, adaptive maybe, but nonetheless independent of any iteration of the algorithm itself; that is, the learning rate at any given iteration of the training process is not determined by the algorithm, but rather by an external scheduler independently.

From a broader view of the gradient-based optimization landscape, this framework (SGD) is in fact only one of several possible choices, albeit it is the easiest to work in practice. A second choice is known as the line search. In this method, after determining the update direction as the gradient direction, the algorithm tries several step sizes in one iteration along the line in this direction (hence the name), and ultimately picks the one step size resulting in the biggest descent. A clear difference here from the SGD method is that the step sizes are chosen within the algorithm itself. Line search would be more costly to execute in practice, but in theory it could provide better training performance than an external learning rate scheduler. TRPO algorithm employs  line search in its iterations to determine the optimal step sizes (in addition to the bounds given by the natural policy gradient theorem).

A third choice is known as the trust region method. For this method, the algorithm no longer separates the decisions on the update direction and update step size; instead, in each iteration, it first formulates a constrained optimization problem, typically with a surrogate objective function that guarantees to improve the "true" objective within the constrained region (the trust region), then attempts to solve this problem (analytically?), obtaining a solution for both the optimal direction and the optimal step size. This is exactly the procedure used by natural policy gradient theorem, and by extension, the TRPO algorithm. And since PPO and its variants could be viewed in theory as simplifications to the TRPO algorithm, it is fair to say that the trust region idea is behind some of the most successful deep RL algorithms today. The trust region methods typically benefit from having excellent theoretical guarantees on their convergence and stability, but suffer from over-complexity in implementation, thus requiring many simplifications to work in practice.

Now we are ready to look in details at the two leading algorithms: TRPO and PPO (clipped version).

#### TRPO = Natural Policy Gradient + Line Search

The main idea behind TRPO<sup>7^</sup> is a re-formulation of the natural policy gradient theorem. By doing so the authors were able to prove theoretically that the algorithm (in its theoretical form) guarantees monotonic policy improvement<sup>11</sup>. The practical algorithm is a combination of natural policy gradient estimator with a line search sub-routine for step size. The detailed practical algorithm is listed as follows:

**Algorithm \| TRPO with line search and Monte Carlo estimation**

1. Initialize $$\pi_0$$ and behavioral policy $$q$$

2. Loop for $$i = 0, 1, 2, ...$$ until convergence, do:

   1. Use the single-path or vine procedures to collect a set of $$(s_t, a_t)$$ pairs along w.t. Monte Carlo estimates of their Q-values $$\hat{Q}(s_t,a_t)$$

   2. Estimate the objective function and KL constraint by averaging over samples and construct the constrained optimization problem as:

      $$\DeclareMathOperator*{\maximize}{maximize}\maximize\limits_\theta\mathbb{E}_{s\sim\rho_{\theta_i},a\sim q}[\frac{\pi_\theta(a \vert s)}{q(a \vert s)}\hat{Q}(s,a)]\;\text{subject to}\;\mathbb{E}_{s\sim\rho_{\theta_i}}[D_{KL}(\pi_{\theta_i} \vert \pi_\theta)]\le\delta$$  (averaging means to treat expectation as $$\frac{1}{N}\sum\limits_{i=1}^N$$)

   3. Construct necessary estimated items for the constrained optimization problem by the following sub-routine:

      1. use sample average to estimate the Fisher Information matrix $$\hat{F}_i=\hat{F}(\theta_i)$$
         1. by using the covariance matrix of policy gradients: $$\hat{F}(\theta_i)=\mathbb{E}_{s\sim\rho_{\theta}}[\nabla_\theta\log\pi_\theta(\cdot \vert s)\nabla_\theta\log\pi_\theta(\cdot \vert s)^T]\vert_{\theta=\theta_i}$$
         2. or, by constructing the Hessian matrix of KL divergence approximately: $$\hat{F}(\theta_i)=\hat{H}(\theta_i)=\mathbb{E}_{s\sim\rho_{\theta}}[\nabla_\theta^2 D_{KL}(\pi_{\theta_i}(\cdot \vert s)\vert\pi_\theta(\cdot \vert s))]\vert_{\theta=\theta_i}$$
      2. estimate the policy gradient by the policy gradient theorem and sample averages: $$\hat{g}=\mathbb{E}_{s\sim\rho_{\theta_i},a\sim p}[\nabla_\theta\log\pi_\theta(a \vert s)\hat{Q}(s,a)]\vert_{\theta=\theta_i}$$
      3. estimate the natural policy gradient step size (also the maximal step size): $$\alpha_i=\sqrt{\frac{2\delta}{\hat{g}^TF_i^{-1}\hat{g}}}$$

   4. Construct the natural policy update: $$\Delta_i=\alpha_i\hat{F}_i^{-1}\hat{g}$$

   5. Perform a line search to determine appropriate step size through Loop for $$j=0,1,2,...L$$, do:

      1. compute update: $$\theta=\theta_i+\beta^j\Delta_i$$ where $$\beta^j$$ is chosen s.t. it exponentially decays the maximal step size $$\alpha_i$$
      2. check two conditions to see if both are true:
         1. the objective function is improved; equivalent to: $$L_{\theta_i}(\theta)=\mathbb{E}_{s\sim\rho_{\theta_i},a\sim q}[\frac{\pi_\theta(a \vert s)}{q(a \vert s)}\hat{Q}(s,a)]\ge0$$
         2. the KL divergence constraint is kept: $$\bar{D}_{KL}=\mathbb{E}_{s\sim\rho_{\theta_i}}[D_{KL}(\pi_{\theta_i} \vert \pi_\theta)]\le\delta$$, where $$D_{KL}=\mathbb{E}_{a\sim \pi_{\theta_i}}[\log\frac{\pi_\theta(a \vert s)}{\pi_{\theta_i}(a \vert s)}]$$
         3. if so then do:
            1. accept the update and set: $$\theta_{i+1}=\theta_i+\beta^j\Delta_i$$
            2. break inner for loop for next iteration

An apparent drawback in the TRPO algorithm is the use of Monte-Carlo estimates for value functions (therefore it has no value network), which could lead to higher variance in practice. The algorithm, being second-order, is also very complicated in its computation.

#### PPO = 1<sup>st</sup>-order TRPO

PPO algorithm<sup>8</sup> could be viewed as an advancement from the TRPO algorithm:

* replaces the constrained optimization problem with second-order solution by an unconstrained problem with first-order solution
* replaces the surrogate objective by a clipped (or penalized) version that is penalized when the step size is too large
* use a more sophisticated estimator for the advantage function to reduce variance
* use a shared value network to better estimate value function
* employ the popular choice of policy entropy as a regularizer to encourage exploration

**Algorithm \| Proximal Policy Optimization w.t. Clipped Surrogate Function in Actor-Critic Style**

1. Initialize shared policy and value parameter $$\theta_0$$

2. Loop for $$i=0,1,2,...$$, do:

   1. Initialize step-counter $$t=1$$

   2. $$t_{start}=t$$ and get initial state $$s_t$$

   3. Loop until $$s_t$$=terminal state or $$t-t_{start}==T$$:

      1. take action $$a_t$$ according to policy $$\pi(a_t \vert s_t;\theta_i)$$ (on-policy)
      2. store reward $$r_t$$ and next state $$s_{t+1}$$
      3. increment $$t$$

   4. Loop for $$t\in1,2,...,T$$, do:

      1. compute advantage estimates using GAE formula: $$\hat{A}_t^{GAE(\gamma,\lambda)}=\sum\limits_{l=0}^\infty(\gamma\lambda)^l\delta_{t+l}^V$$

   5. Construct the objective function through the following sub-routine by averaging over the mini-batch:

      (here $$\hat{\mathbb{E}_t}=\frac{1}{T}\sum\limits_{t=0}^{T-1}$$ is an approximation to the true expectation, $$\mathbb{E}_{s\sim\rho_{\theta_i},a\sim \pi_{\theta_i}}$$)

      1. compute the clipped surrogate loss: $$L^{CLP}(\theta)=\hat{\mathbb{E}}_t[\min(\frac{\pi(a_t \vert s_t;\theta)}{\pi(a_t \vert s_t;\theta_i)}\hat{A}_t^{GAE(\gamma,\lambda)}, \text{clip}(\frac{\pi(a_t \vert s_t;\theta)}{\pi(a_t \vert s_t;\theta_i)}, 1-\epsilon,1+\epsilon)\hat{A}_t^{GAE(\gamma,\lambda)}]$$
      2. compute the value function squared loss: $$L^{VF}(\theta)=\hat{\mathbb{E}}_t[(V(s_t;\theta_i) - \hat{V}(s_t))^2]$$, where $$\hat{V}(s_t)=\sum\limits_{l=0}^{T-1}\gamma^lr_{t+l}$$ is the estimate for value target
      3. compute the entropy of policy function: $$S(\theta)=\hat{\mathbb{E}}_t[\pi(a_t \vert s_t;\theta)\log(\frac{1}{\pi(a_t \vert s_t;\theta)})]$$
      4. construct the objective function as the weighted sum of the three terms: $$L(\theta)=L^{CLP}(\theta)-c_1L^{VF}(\theta)+c_2S(\theta)$$

   6. Update $$\theta$$ by a gradient-based optimizer (e.g., SGD, Adam) w.r.t. to $$L(\theta)$$; set $$\theta_{i+1}=\theta$$

The main motivation behind PPO is to remove the KL-divergence constraint so that the algorithm could become first-order, without hurting too much of the stability. Without the constraint, the original TRPO objective may lead to excessively large policy update. By clipping the objective, it penalizes moving too far from $$\theta_{old}$$ in one update. This penalty could also be achieved by using a KL-divergence penalty term in the objective function directly; the resulting algorithm is still first-order. The latter idea is further explored in the DPPO-penalty paper<sup>9</sup> in distributed settings.

#### A closer look at PPO and TRPO

In ICLR 2020, Andrew Ilyas et al<sup>12</sup> shed more light on the correlation between the empirical behaviors of deep policy gradient algorithms and their theoretical expectations. To this end, the authors conducted experiments on continuous domains (using the MuJoCo physics engine) for PPO and TRPO algorithms. Their analysis rests upon the following key methodologies:

* use the statistics on a large number of samples (up to ~1M) in terms of state-action pairs in a trajectory to represent the "true" quantities, such as the value function and the policy gradient
* use averaged pair-wise cosine similarities within a batch of independent rollouts as a measure for the variance
* use averaged cosine similarity or mean relative error (MRE) between the current iteration and the "true" quantity as a measure for convergence
* plot the reward-parameter and surrogate-parameter landscapes to observe whether improvements in the surrogate objective lead to improvements in the reward landscape

The paper investigated the various aforementioned characteristics that distinguish a policy gradient algorithm. The results are quite astonishing:

* At typical sample complexity (~2,000 pairs in a trajectory), the gradient estimators often poorly approximates the "true" gradient; moreover, the variance in gradient estimation in fact becomes larger instead of smaller as training progresses. This means that these successful algorithms actually operate with increasingly poor gradient estimation, while being able to improve their performance on the benchmarks
* The use of a value network only marginally reduces the variance of the gradient estimator, compared to use Monte-Carlo returns. This is the argument made in the PPO vs. TRPO case earlier, which is very counter-intuitive from theoretical perspective, and also doesn't correlate well with the fact that the variance reduction techniques often translate to moderate to significant performance on the benchmarks.
* Sometimes an update step that improves the surrogate function used by TRPO or PPO could actually hurt the "true" reward landscape.

This paper provided interesting observations that warrant close follow-up's. At a glance, the methodologies used by the authors seemed reasonable. Out of the several observations, the most significant is perhaps the first one: gradient estimation is very poor during training. If this is the case, then a bottleneck for deep policy gradients would be the sample complexity. However, the reinforcement learning algorithms in general already suffer from long training time (or simulation time), a more stringent requirement on the length of the sampled trajectories would only make the training process prohibitively costly. This study also suggested further research could be done (possibly in ablation study manner) to investigate the correlation between performance on the benchmarks versus performances of the various characteristics of a deep policy gradient algorithm. At the same time, it raised questions as to whether the seemingly sound theories and "justifications" (albeit often based on unrealistic assumptions in practical algorithms, such as assuming access to "true" value functions) could actually explain why some of the algorithms perform well on benchmarks; perhaps future RL research could focus less on crafting a solid theoretical footing with complex assumptions than on comprehensive, empirical studies.

#### Reference

1. Volodymyr Mnih et al, "[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)", ICML 2016
2. Ziyu Wang et al, "[Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)", ICLR 2017
3. Volodymyr Mnih et al, "[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)", NeurIPS 2013
4. Richard Sutton et al, "[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)", NeurIPS 2000
5. John Schulman et al, "[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)" , ICLR 2016
6. Sham Kakade, "[A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)", NeurIPS 2002
7. John Schulman et al, "[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)", ICML 2015
8. John Schulman et al, "[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)", Arxiv 2017
9. Nicolas Heess et al, "[Emergence of Locomotion Behaviors in Rich Environments](https://arxiv.org/abs/1707.02286)", Arxiv 2017
10. Samuel L. Smith et al, "[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)", ICLR 2018
11. Sham Kakade & John Langford, "[Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)", ICML 2002
12. Andrew Ilyas et al, "[A Closer Look at Deep Policy Gradients](https://openreview.net/forum?id=ryxdEkHtPS)", ICLR 2020