---
layout: post
comments: true
title: Convolutional Neural Tangent Kernels
date: 2021-05-08
author: Baihua Xie
tags: neural-tangent-kernels theory
mins: 30-min
---

Since proposed by Jacob el al<sup>1</sup> in 2018, _neural tangent kernels_ have become an important tool in recent literature<sup>2,3,4</sup> that study the theoretical behaviors of deep neural networks. In this post I'll discuss one of the earlier works on convolutional neural tangent kernels, or CNTK, by Arora et al<sup>5</sup> in NeurIPS 2019. In this paper, the authors derived an efficient algorithm that computes a CNTK kernel regression problem which the authors claim could estimate the training dynamics of ReLU CNN's trained with gradient descent. The CNTK algorithm achieves 77.4% accuracy on the CIFAR-10 dataset. Although a poor performance compared to well-trained deep CNN's, this is nonetheless an impressive +10% higher than previous state-of-the-art for a purely kernel-based method. Since the main significance of NTKs lies with them being a powerful tool to study the dynamics of training deep neural networks, one could argue that the absolute accuracy numbers are of less concern. In this post I'll focus on how the CNTK works exactly, as well as what kind of assumptions and limitations it might have for studying mainstream deep neural networks.

First let's get a few preliminaries out of the way:

**Kernel Regression.** Let $$f(\theta;x)\in\mathbb{R}$$ denote the output of a neural network given weights $$\theta$$ and an input sample $$x$$. Let $$X$$ and $$X^*$$ represent the set of training and testing dataset, respectively. A regression or classification problem would ask the network to predict $$f(\theta; x^*)$$ given the training samples $$\{X,y\}$$. Kernel regression solves this problem by giving a closed-form expression for the prediction as the following:
$$
f(\theta;x^*)=\sum_j^N\mathcal{k}(x^*,x_j)y_j
$$

Here, $$\{x_j,y_j\}\in\{X,y\}$$ are the $$N$$ training samples; $$\mathcal{k}(\cdot,\cdot)$$ is a kernel function between two samples that produces a similarity score. In essence, kernel regression produces a prediction for an unseen sample in the test set as a weighted combination of the labels in the training set.

**Neural Tangent Kernel (NTK).** Jacob et al<sup>1</sup> proposed specific forms of the kernel functions $$\mathcal{k}(\cdot,\cdot)$$ for different deep neural networks, such that the kernel regression solution with these kernels can be shown to be (approximately) equivalent to the results produced by these neural networks trained using gradient descent. Specifically, for weakly-trained networks, i.e., networks with fixed weights except for the top layer trained with $$\ell_2$$ loss, the following kernel function was derived:
$$
\mathcal{k}(x,x')=\mathbb{E}_{\theta\sim W}[f(\theta;x)f(\theta;x')]
$$

where the expectation is taken for the weights following its initialization distribution, $$W$$, which is usually Gaussian in practice.

For fully trained networks using gradient descent with $\ell_2$ loss, the following kernel was proposed, referred to as the _neural tangent kernels_ or NTKs:

$$
\text{NTK}(x,x')=\mathbb{E}_{\theta\sim W}[<\frac{\partial{f(\theta;x)}}{\partial{\theta}}, \frac{\partial{f(\theta;x')}}{\partial{\theta}}>]
$$

Note that this kernel is not in closed-form, so it would require further approximations to be computed in practice. But let's first see how to arrive at this kernel (proof of Lemma 3.1 in the Arora paper). We are interested in the training dynamics $$\theta(t)$$, i.e., the evolution of the weights w.r.t. time as the network is trained, since the final prediction is simply $$f(\theta(T);x^*)$$ where $$T$$ stands for the end of training. The networks is trained with gradient descent, so we have $$\theta(t+\Delta t)=\theta(t)-\nabla_\theta\ell(\theta(t))\cdot\Delta t$$; of course in practice the training is actually done iteratively, but the training dynamics can still be expressed in a differential equation:

$$
\frac{d\theta(t)}{dt}=-\nabla_\theta\ell(\theta(t))=-\sum_j^N(f(\theta(t);x_j)-y_j)\frac{\partial f(\theta(t);x_j)}{\partial\theta}
$$

This makes use of the $$\ell_2$$ loss: $$\frac{1}{2}\sum_j^N(f(\theta;x_j)-y_j)^2$$. We can further have:

$$
\begin{split}
\frac{\partial f(\theta(t);x_i)}{\partial t}&=\frac{\partial f(\theta(t);x_i)}{\partial \theta(t)}\frac{d\theta(t)}{dt}\\
&=-\sum_j^N(f(\theta(t);x_j)-y_j)<\frac{\partial f(\theta(t);x_i)}{\partial\theta},\frac{\partial f(\theta(t);x_j)}{\partial\theta}>
\end{split}
$$

Here $$<\cdot,\cdot>$$ denotes the inner product between two vectors (the partial derivatives for the gradients are vectors); this derivation can work because the first term on the R.H.S. is actually a scalar value, so it can be moved outside of the dot-product. If we consider all the samples in the training set, the dot products would be extended to a matrix $$H(t)$$ such that the entries are:

$$
H(t)_{ij}=<\frac{\partial f(\theta(t);x_i)}{\partial\theta},\frac{\partial f(\theta(t);x_j)}{\partial\theta}>=\mathcal{k}(x_i,x_j)
$$

Let $$u(t)=f(\theta(t);x_i)_{i\in N}$$ denote the vector of all training outputs and let $$y$$ denote all training labels, if we group the dynamics for all training samples together it can be expressed as:

$$
\frac{du(t)}{dt}=-H(t)(u(t)-y)
$$

which is a classic dynamics of kernel regression _with gradient flow_. We make an important assumption here: the network's width $$w$$ goes to $$\infty$$. For the NTK matrix $$H(t)$$, this assumption leads to two theoretical results:

$$
\begin{split}
&1) \lim_{w\to\infty}H(t)=H(0)\\
&2) \lim_{w\to\infty}(H(0))=H'
\end{split}
$$

The first result shows that, at the inftyite width limit, the NTK matrix _remains constant_ as the initialized matrix $$H(0)$$ during training. This is a very surprising result, as it essentially says that the parameter-pass in backpropagation is actually fixed during the entire training process; if this is true then theoretically it could drastically reduce the complexity of backpropagation. It would be very interesting to see whether any practical, finite-width neural networks exhibit training behaviors even only close to this analysis.

The second result further simplifies the computation of the NTK matrix to be only evaluated on the training set, denoted by $$H'$$. With these two results, we would only need to compute the NTK matrix once after weight initialization for all the samples in the training set; the training process would not need to update the NTK matrix.

The closed-form solution for the kernel regression dynamics is:

$$
f(\theta;x^*)=\lim_{t\to\infty}f(\theta(t);x^*)=(\mathcal{k}(x^*,x_1),...,\mathcal{k}(x^*,x_N))(H')^{-1}y
$$

In summary, by assuming the network's width can go to infinity, the NTK formulation proposed that, instead of having to train the neural network using gradient descent and backpropagation, one can simply compute the predicted outputs in the following steps:

1. Initialize the weights;
2. Compute the NTK matrix $$H'$$ for all training samples;
3. With each new test sample $$x^*$$, compute the NTK kernel score of $$\mathcal{k}(x^*,x_i)$$ w.r.t. each sample in the training set;
4. Use the closed-form solution to compute the predicted output;

At a glance, this formulation goes against the fundamental ideas behind representation learning, which is what gives the expressive powers to deep neural networks. One can observe that this procedure has several serious problems:

1. The weights are only initialized, never updated; if this is the case then why do we need the weights to begin with? also where does the expectation w.r.t. initialization distribution go during the above derivation of NTKs? 
2. For every new test sample we need to re-evaluate the solution on the entire training set; this can have huge computational complexity implications, as the training set in practice is usually very large;
3. Can this solution overfit the training set to begin with? i.e., if I pass a training sample to the formula, is it guaranteed to output the ground truth?

**Gaussian process (GP) in neural networks.** NTK is closely related to the concurrently developed GP view of infinitely wide neural networks, pioneered by Lee et al<sup>6</sup> in 2018. GP is a non-parametric method to model probability distribution of _functions_ instead of _variables_. In classic terms, we say that a function $$f\sim\text{GP}(\mu,\Sigma)$$ if for any subset of the function's values conditioned on its inputs, $$\{f(x_1),f(x_2),...,f(x_k\}$$, the joint probability distribution of this subset follows a Gaussian distribution of $$\sim N(\mu,\Sigma)$$. In practice usually the GP's have zero mean, referred to as _centered GP_; GP is thus fully characterize by its covariance matrix (which can be the covariance or any kernel function in practice). Since the covariance is completely determined given the training set, GP is a non-parametric model.

In the context of a neural network, following Lee et al, if we let the pre-nonlinearity activations be denoted as $$f^{(h)}(x)$$ and the post-nonlinearity activations be denoted as $$g^{(h)}(x)$$ for each layer (note that this views one layer in the network as nonlinearity first followed by affine transformation); here $$x$$ emphasizes that the activations are _conditioned_ on an input sample $$x$$. Lee et al conjectured that, in the limit of $$d_h\to\infty$$, due to the Central Limit Theorem and the assumption that $$x$$ are i.i.d. samples, the activations at each layer should approximate a random variable that follows a Gaussian distribution with centered mean (=0). What's really interesting is their next hypothesis: if we treat the activations conditioned on different input samples as different random variables, then any subset of $$\{f^{(h)}(x_{\alpha=1}),...,f^{(h)}(x_{\alpha=k})\}$$ would follow a centered _multivariate_ Gaussian distribution; this is exactly the definition of a Gaussian Process, i.e., we can model the activations in the neural network by $$f^{(h)}(x)\sim\text{GP}(0,\Sigma^{(h)})$$. Just like NTK, this is another method to study the theoretical behaviors of deep neural networks (under infinite width limit) in _closed-form_.

To calculate the covariance matrix, we need to only look at any two samples from the training set and evaluate $$\Sigma^{(h)}(x,x')$$. Note that the $$x$$ here only means that the calculations are _conditioned_ on these inputs; the actual random variables of which the covariance is evaluated are of course $$f^{(h)}(x)$$ and $$f^{(h)}(x')$$. Using the basic property that $$\text{Var}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$$ and the assumption that both random variables are centered, we obtain:

$$
\begin{split}
\Sigma^{(h)}(x,x')&=\mathbb{E}[f^{(h)}(x)f^{(h)}(x')]\\
&=c_\phi\mathbb{E}[g^{(h)}(x)g^{(h)}(x')]\\
&=c_\phi\mathbb{E}_{f^{(h-1)}\sim\text{GP}(0,\Sigma^{(h-1)})}[\sigma(f^{(h-1)}(x))\sigma(f^{(h-1)}(x'))]
\end{split}
$$

Here we have treated each layer as consisting of a non-linearity followed by an affine transformation; for details check Lee et al Eqn. (4). $$c_\phi$$ is a constant, in their paper it is derived to be the standard deviation of the layer's weight matrix, but in Arora's paper the definition seems to be different; here I just take it to be a general value since it's not important for this discussion. $$\sigma(\cdot)$$ is the nonlinearity. The critical part in the final expression is the expectation; it is taken over the distribution of the previous layer's output (which is the input to current layer), which we know to be a centered GP with covariance $$\Sigma^{(h-1)}$$. This thus forms a recursive structure; we can now calculate the covariance matrix for any two input samples $$x$$ and $$x'$$ layer-by-layer iteratively as follows:

1. Initialize $$\Sigma^{(0)}(x,x')$$; there are several initialization schemes discussed in relevant papers, e.g., just zero, or take dot-product between inputs $$x^Tx'$$;
2. For each layer $$h$$ staring from 0, compute the covariance matrix that governs the GP at this layer; for any two input samples the covariance matrix is 2x2 and symmetrical (necessarily?), with entries: $$\Sigma^{(h)}(x,x')$$,$$\Sigma^{(h)}(x,x)$$ and $$\Sigma^{(h)}(x',x')$$; let's denote this matrix as $$\Lambda$$;
3. Draw two random variables $$u,v\sim N(0,\Lambda)$$ and compute the expectation on the last line of the equation above; In practice expectation can be approximated by tricks like Monte-Carlo sampling; in Arora paper the authors proposed a smart trick to compute the expectation more efficiently and accurately;
4. Repeat 2-3 until $$h=L$$;

For step 2., one question is not very clear to me from Lee et al: the expectation should in fact be taken over the joint multivariate Gaussian characterized by the covariance matrix conditioned on _all_ input samples; however here we are only counting the two input samples for the covariance (so the matrix is $$2\times2$$ instead of $$N\times N$$, where $$N$$ is number of samples), ignoring other samples; is there a particular reason for this simplification?

In summary, the essence of the GP view of neural networks is an iteratively procedure to build approximations to the network's outputs. This procedure was adapted in the NTK formulation to be used to compute the NTK kernels.

We have seen three implications that follows directly from the assumption that the network's width goes to infinity:

* the NTK matrix $$H(t)$$ is invariant and equal to $$H(0)$$ during training;
* $$H(0)$$ can be evaluated by only samples from the training set; these two results lead directly to the kernel regression solution of NTK;
* and now we have that the activations in the layers follow a Gaussian Process fully characterized by the covariance matrix $$\Sigma^{(h)}$$;

For now let's turn to the main problem being addressed in the Arora paper. Since the NTK is not a closed-form function, approximations would be required to use it in practical networks. Here I'll follow the derivation of an approximated NTK for networks comprised of only fully connected (or feed-forward) layers with ReLU activations (because it's relatively easier to work out...). The NTK for CNN's follows very similar derivations, as CNN's are essentially fully connected networks sparsified through weight sharing.

##### Neural Tangent Kernels for Fully Connected Networks

For NTK, we need to evaluate the dot product term:

$$
<\frac{\partial f(\theta;x)}{\partial\theta},\frac{\partial f(\theta;x')}{\partial\theta}>
$$

$$x,x'$$ are any two samples in the training set; for simplicity I have dropped the index $$t$$. Define the fully-connected network with $$L$$ layers as follows:

$$
\begin{split}
f^{(h)}(x)&=W^{(h)}g^{(h-1)}(x)\\
g^{(h-1)}&=\sigma(f^{(h-1)(x)})
\end{split}
$$

Here $$h=1,...,L$$ is the layer numbers, $$\sigma(\cdot)$$ is the activation function (taken as ReLU in this paper). I have dropped the scaling terms before the activation function and merge the bias into weights for simplicity. Now the weights are really $$\theta=\{W^{1},W^{2},...\}$$. Since the dot-product is just a sum, we can re-write it as:

$$
<\frac{\partial f(\theta;x)}{\partial\theta},\frac{\partial f(\theta;x')}{\partial\theta}>=\sum_{h=1}^{L}<\frac{\partial f(\theta;x)}{\partial W^{(h)}},\frac{\partial f(\theta;x')}{\partial W^{(h)}}>
$$

We need to clarify something first. Although on the R.H.S., the terms in the dot-product would be a matrix $$\in\mathbb{R}^{d_{h}\times d_{h-1}}$$, we actually treat it as a "flattened" matrix when performing the dot-product (i.e., in code we would write something like ```x.view(-1)```). The result of the dot-product would be a scalar value. This is because in the derivation of NTK, we treat all the weights in the network $$\theta$$ as a vector; now in multi-layered network the weights are matrices or tensors at each layer for practical purposes, but for theoretical purposes it is more convenient to view them as flattened. 

Now use the chain rule to expand the term on the R.H.S. from the output to layer $$h$$:

$$
\frac{\partial f(\theta;x)}{\partial W^{(h)}}=\frac{\partial f^{(h+1)}(x)}{\partial f^{(h)}(x)}\cdot\cdot\cdot\frac{\partial f^{(L)}(x)}{\partial f^{(L-1)}(x)}\frac{\partial f(\theta;x)}{\partial f^{(L)}(x)}\frac{\partial f^{(h)}(x)}{\partial W^{(h)}}\in\mathbb{R}^{d_h\times d_{h-1}}
$$

(I have re-ordered the terms to reflect the matrix multiplication orders properly; the derivatives assume denominator layout and vectors assume column-major; see below). Use the definition of the fully-connected layers:

$$
\frac{\partial f^{(h+1)}(x)}{\partial f^{(h)}(x)}=\text{diag}(\dot{\sigma}(f^{(h)}(x)))(W^{(h+1)})^T
$$

Let's check the dimensions here: $$f^{(h+1)}\in\mathbb{R}^{d_{h+1}}$$ and $$f^{(h)}\in\mathbb{R}^{d_{h}}$$ are column vectors; if we consider using the denominator layout, the L.H.S. should be a matrix $$\in\mathbb{R}^{d_{h}\times d_{h+1}}$$; on the R.H.S., the diagonal matrix is $$\in\mathbb{R}^{d_{h}\times d_{h}}$$ and the transposed weight matrix is $$\in\mathbb{R}^{d_{h}\times d_{h+1}}$$; the dimensions match under denominator layout.

The last term would be evaluated as:

$$
\frac{\partial f^{(h)}(x)}{\partial W^{(h)}}=I\otimes g^{(h-1)}(x)^T=g^{(h-1)}(x)^T
$$

The dimensions also match: L.H.S. is a vector-by-matrix derivative, the complete form would require using the tensor product $$\otimes$$ with the identity matrix; but since it is actually a linear layer, it can be shown that the result would be reduced to a row vector $$\in\mathbb{R}^{1\times d_{h-1}}$$; so we denote the R.H.S. with transpose to make it a row vector.

If we define recursively for each layer $$h$$ such that:

$$
b^{(h)}(x)=\text{diag}(\sigma(f^{(h)}(x)))(W^{(h+1)})^T\cdot b^{(h+1)}\in\mathbb{R}^{d_{h}\times 1}\hspace{10pt}\text{for}\hspace{2pt}h=h,h+1,...,L
$$

we can write the final expression in a compact form as:

$$
\frac{\partial f(\theta;x)}{\partial W^{(h)}}=b^{(h)}(x)\cdot g^{(h-1)}(x)^T
$$

The $$\cdot$$ represents matrix multiplication. If we take away the transpose on $$g$$, then this is an outer product. (Need to check this transformation later) The NTK entry can be re-written as:

$$
<\frac{\partial f(\theta;x)}{\partial W^{(h)}},\frac{\partial f(\theta;x')}{\partial W^{(h)}}>=<g^{(h-1)}(x),g^{(h-1)}(x')>\cdot<b^{(h)}(x), b^{(h)}(x')>\in\mathbb{R}
$$

In evaluating the first term on the R.H.S., the authors made a very strange connection from dot product to expectation. As in their eqn. (16) in section D, essentially they casually equated the dot-product and the covariance of the post-activations $$g(x), g(x')$$, so they have:

$$
<g^{(h-1)}(x),g^{(h-1)}(x')>=\mathbb{E}[g^{(h-1)}(x)g^{(h-1)}]=\mathbb{E}[f^{(h)}(x)f^{(h)}]=\Sigma^{(h)}(x,x')
$$

All the equalities above except for the first one have been discussed previously. For the first equality, the only source of reference I could find is that, the authors might be treating the activations $$g^{(h)}$$ as the _vector representation_ of a univariate random variable. To treat the activations as a single random variable seems actually consistent with the GP view introduced previously; there Lee et al treated $$g^{(h)}(x)$$ to be a random variable following a Gaussian distribution, and the activations conditioned on different input samples to be different random variables, thus together they form a _random vector_ that follows a _multivariate_ Gaussian distribution. However Lee et al never used dot-products on these random variables.

There exists a curious link between probability theories (about random variables) and linear algebra (about vectors and matrices): a vector can also be viewed as a form of representation for a random variable. This seems odd at first, but in essence random variables are really neither random nor themselves variables. In fact, they are really _measurable_ functions defined on a vector space which maps from an event space to a real-valued probability space. I won't go into details in this post as this concerns the measure theory; but a simple analogy is that we can treat the index $$i$$ of a vector of dimension $$n$$ to be having a probability of "occurrence" equal to $$1/n$$. In practice this view of random variables give rise to certain interesting geometrical interpretations of probability, such as the expectation being a projection onto the basis vector, etc.; but another useful result with this view is that it allows us to use the tools in linear algebra along with probability.

If the above thinking is correct, then we have a method to iteratively evaluate the dot-product on activations. Now we need to evaluate the dot-product on $$b^{(h)}$$. The procedure is relatively straight-forward; one can refer to section D for more details. The authors invoked the same vector interpretation here as well; they arrived at:

$$
<b^{(h)}(x), b^{(h)}(x')>=\prod_{h'=h}^L\dot{\Sigma}^{h'}(x,x')
$$

To arrive at this simple form, the authors had to invoke a trick to decouple $$W^{(h)}$$ and $$b^{(h)}(x)$$, which are interdependent. It is assumed that we can replace $$W^{(h)}$$ with a new sample $$\tilde{W}^{(h)}$$ without changing its limit; when made rigorous, this can be a useful trick in general. Here the derivation of the covariance matrix means that when we take the expectation in the iterative GP formula, we are computing $$c_\phi\mathbb{E}_{f^{(h-1)}\sim\text{GP}(0,\Sigma^{(h-1)})}[\dot{\sigma}(f^{(h-1)}(x))\dot{\sigma}(f^{(h-1)}(x'))]$$ instead.

Finally, we have the expression for the NTK kernel entries:

$$
H(x,x')=<\frac{\partial f(\theta;x)}{\partial\theta},\frac{\partial f(\theta;x')}{\partial\theta}>=\sum_{h=1}^{L}(\Sigma^{h}(x,x')\prod_{h'=h}^L\dot{\Sigma}^{h'}(x,x'))=\Theta^{(L)}(x,x')
$$

with the covariances at each layer evaluated by the same GP iterative procedure.

##### Convolutional Neural Tangent Kernels

Arora et al made two primary contributions in their paper. Firstly, they derived a closed-form expression of NTK for ReLU-activated CNN's with global average pooling, termed CNTK; this expression is also based on the covariance matrices from the GP view, and is evaluated iteratively layer-by-layer in a similar fashion to NTK for fully connected networks discussed above. Secondly, seeing the computational cost of estimating the expectation in the covariance iterations, the authors provided a customized CUDA kernel to efficiently carry out an approximate version of the computation through a neat matrix trick.

Their derivation for CNTK formula is surprising complicated, given that one would assume that CNN's are just weight-shared fully connected networks. I'm not entirely sure of the theoretical significance of their derivation, so I'll skip it in this post and discuss its results directly. For CNN's, we define $$x,x'$$ to be two input images with dimensions $$\in\mathbb{R}^{P\times Q}$$. We define all computations about covariances on a single patch $$\mathcal{D}_{ij,i'j'}\in[P]\times[Q]\times[P]\times{Q}$$ that equals in size the receptive field $$q\times q$$ of the convolutional kernels (so we'll subscript all the quantities with $$_{ij,i'j'}$$). For each layer $$h$$ in the network, assume the the filter channels are $$\alpha=1,2,...,C^{(h)}$$. We define a quantity $$K^{(h)}_{(\alpha)}(x,x')_{ij,i'j'}\in\mathbb{R}^{q\times q\times q\times q}$$ for each patch at each channel in layer $$h$$, as well as a quantity $$\Sigma^{(h)}(x,x')_{ij,i'j'}\in\mathbb{R}$$ for each patch at all channels of the same layer; these quantities serve the same purpose as the covariance matrices in the NTK formulation and are iteratively updated.

Their iterative CNTK formula to produce the output estimate at the final layer is as follows:

1. Initialize the NTK entries at the input $$\Theta^{(0)}(x,x')\in\mathbb{R}$$; for vanilla CNN's the authors opted to initialize as $$\Sigma^{0}(x,x')=x^Tx'$$, same as before, while for CNN's with global average pooling it is initialized as simply $$0$$; I'm not sure of the theoretical thinking behind this difference;

2. Initialize $$K^{(0)}_{(\alpha)}(x,x')_{ij,i'j'}=x_{(\alpha)ij,i'j'}\otimes x'_{(\alpha)ij,i'j'}\in\mathbb{R}^{q\times q\times q\times q}$$; here $$\otimes$$ is tensor product or out;

3. Initialize $$\Sigma^{(0)}(x,x')_{ij,i'j'}=\sum_{\alpha=1}^{C^{(0)}}\text{tr}(K^{(0)}_{(\alpha)}(x,x')_{ij,i'j'})\in\mathbb{R}$$; this is a scalar value; $$\text{tr}(\cdot)$$ is the trace operator;

4. For each layer $$h=1,2,...,L$$ and for each patch $$\mathcal{D}_{ij,i'j'}\in[P]\times[Q]\times[P]\times[Q]$$ at that layer, compute the covariance matrix $$\Lambda^{(h)}(x,x')_{ij,i'j'}\in\mathbb{R}^{2\times2}$$ used in GP, with the kernel values from last layer $$\Sigma^{(h-1)}(x,x')_{ij,i'j'}$$, $$\Sigma^{(h-1)}(x,x)_{ij,i'j'}$$, and $$\Sigma^{(h-1)}(x',x')_{ij,i'j'}$$, same as before;

5. Assume that the pre-nonlinearity activations follow a centered GP, use the GP formula to update $$K^{(h)}_{ij,i'j'}$$ and its derivative  $$\dot{K}^{(h)}_{ij,i'j'}$$ by expectation:

   $$
   \begin{split}
   K^{(h)}(x,x')_{ij,i'j'}&\gets \frac{c_\phi}{q^2}\cdot\sum_{u,v\sim N(0,\Lambda^{(h)}(x,x')_{ij,i'j'})}
   [\sigma(u)\sigma(v)]\in\mathbb{R}^{q\times q\times q\times q}\\
   \dot{K}^{(h)}(x,x')_{ij,i'j'}&\gets \frac{c_\phi}{q^2}\cdot\sum_{u,v\sim N(0,\Lambda^{(h)}(x,x')_{ij,i'j'})}
   [\dot{\sigma}(u)\dot{\sigma}(v)]\in\mathbb{R}^{q\times q\times q\times q}
   \end{split}
   $$

   Note that the quantities are no longer defined on each filter channel (the subscript $$_\alpha$$ is removed). The reason is not explicitly explained in the paper, but one would assume that, since the covariance matrix is defined over all channels, the expectation should produce a quantity that is equivalent across channels as well.

6. Compute the CNTK entry for each patch at layer $$h$$ as:

   $$
   \Theta^{(h)}(x,x')_{ij,i'j'}=\text{tr}([\dot{K}^{(h)}(x,x')_{ij,i'j'}\odot\Theta^{(h-1)}(x,x')_{ij,i'j'}]+K^{(h)}(x,x')_{ij,i'j'})\in\mathbb{R}
   $$

   here $$\odot$$ denotes element-wise multiplication since $$\Theta$$ is a scalar value;

7. Update $$\Sigma^{(h)}(x,x')_{ij,i'j'}=\text{tr}(K^{(h)}(x,x')_{ij,i'j'})\in\mathbb{R}$$;

8. Repeat steps 4-7 until $$h=L$$. At the last layer, we obtain $$\Theta^{(L)}(x,x')_{ij,i'j'}$$ using the same formula in step 6 for vanilla CNN's; for CNN's with global average pooling, the authors chose to drop the $$K^{(L)}(x,x')_{ij,i'j'}$$ term;

9. Group the CNTK entries for all patches into a single tensor $$\Theta^{(L)}(x,x')\in\mathbb{R}^{P\times Q\times P\times Q}$$ and assume zero-padding;

10. Compute the final CNTK value;

    for vanilla CNN's we take the trace of the diagonal elements: (Q: how do we take trace over tensors?)

    $$
    H(x,x')=\text{tr}(\Theta^{(L)}(x,x'))
    $$

    for CNN's with global average pooling, the authors apply a similar "pooling" operation over the $$\Theta$$ matrix by taking the global average:
    
    $$
    H(x,x')=\frac{1}{P^2Q^2}\sum_{ij,i'j'}\Theta^{(L)}(x,x')_{ij,i'j'}
    $$

The computational challenge of the above procedure is at step 5, when the expectation needs to be approximated. Instead of traditional techniques like MC sampling, the authors came up with a pretty creative solution. They observed that the following closed-form _exact_ expectations are valid:

$$
\begin{split}
\sum_{u,v\sim N(0,D\Lambda D)}[\sigma(u)\sigma(v)]&=\frac{\lambda(\pi-\arccos(\lambda))+\sqrt{1-\lambda^2}}{2\pi}\cdot c_1c_2\\
\sum_{u,v\sim N(0,D\Lambda D)}[\dot{\sigma}(u)\dot{\sigma}(v)]&=\frac{\pi-\arccos(\lambda)}{2\pi}
\end{split}
$$

if the covariance matrix $$A=D\Lambda D$$ such that $$\Lambda=\begin{pmatrix}
1 & \lambda \\
\lambda & 1
\end{pmatrix}$$ and $$D=\begin{pmatrix}
c_1 & 0 \\
0 & c_2
\end{pmatrix}$$. We know the entries in the covariance matrix $$A=\begin{pmatrix}
\Sigma_{xx} & \Sigma_{xx'} \\
\Sigma_{x'x} & \Sigma_{x'x'}
\end{pmatrix}$$; then we can obtain the following calculations:

$$
\begin{split}
c_1^2&=\Sigma_{xx}\\
c_2^2&=\Sigma_{x'x'}\\
\lambda&=\frac{\Sigma_{xx'}}{c_1c_2}
\end{split}
$$

Note that we have assumed the covariance matrix is symmetrical, which generally should be true; also since we need to take square roots, this means the entries in the covariance matrix must be the result of a non-negative kernel function. With these relatively general assumptions, the authors have been able to reduce the expectation computation into a simple, closed-form evaluation. An interesting note: doesn't $$\lambda$$ looks like a correlation?

The following codes are snippets from the [source code](https://github.com/ruosongwang/CNTK) published along with the paper; I added some comments.

```python
# CUDA kernel for activation
void trans(float s[32][32][32][32], float t[32][32][32][32], const float l[32][32], const float r[32][32], const float il[32][32], const float ir[32][32])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);

    # S = K(h-1), T = \Theta(h-1);
    # L = c1, R = c2, iL = 1/c1, iR = 1/c2 => the matrix D in section I in the paper;
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
    # S = \lambda;
	S = S * iL * iR;

	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f)))*L*R/28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T * S + BS;
	s[x1][y1][x2][y2] = BS;
}
```



In the CUDA kernel, `s` and `t` stores the current values for $$K^{(h)}(x,x')_{ij,i'j'}$$ and $$\Theta^{(h)}(x,x')_{ij,i'j'}$$; notice their tensor shapes are both `32x32x32x32`, because the experiments are conducted on CIFAR-10 dataset with image size = 32x32. The expectations are computed using the fast formula above. After this kernel, we also need to compute the trace on `s` to obtain the current values for $$\Sigma^{(h)}(x,x')$$; this is done by the following customized convolution kernel:

```python
# CUDA kernel for convolution operation -> trace operation
void conv3(const float s[32][32][32][32], float t[32][32][32][32])
{
    # x1=batch; y1=channel; x2,y2 = H/W;
    # each thread block has (32, 32) threads = 1 feature map;
    # -31 b/c there are (63, 63) blocks in the grid, but only batch=32 and channels=32 are used;
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	__shared__ float d[32 + 2][32 + 2];

    # padding;
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}
	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();
    # take trace;
	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];
}
```



However it is not very clear to me how this kernel, which looks just like a normal 3x3 conv filter, performs the trace operation over $$K^{(h)}(x,x')_{ij,i'j'}$$ and $$\Theta^{(h)}(x,x')_{ij,i'j'}$$. The authors call these two kernels in the iterative algorithm to compute the CNTK values $$H(x,x')$$:


```python
# computes H(x,z)
# here Lx=c1, lZ=c2, iLx=1/c1, iLz=1/c2; the variables are lists containing all such diagonal values for all input samples;
def xz(x, z, Lx, Lz, iLx, iLz):
    # initialize K = step 2;
	S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
    # initialize \Sigma by taking trace of K = step 3;
	conv3(conv_blocks, conv_threads, (S, S))
    # initialize \Theta = step 1;
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
    # if not fix means it's vanilla CNN, initialize \Theta same as K using dot products;
	if not fix:
		T += S

	for i in range(1, d - 1):
        # compute expectations and update K, K', \Theta (before trace) = step 5;
		trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))
        # update \Sigma by taking trace of updated K = step 7;
		conv3(conv_blocks, conv_threads, (S, S))
        # update \Theta by taking trace of updated \Theta (before trace) = step 6;
		conv3(conv_blocks, conv_threads, (T, T))

    # final \Theta after trace;
	trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))	

    # if fix means use global average pooling; the authors dropped the K term as in step 8;
	if fix:
		T -= S
    # \Theta(h) = trace(~\Theta(h-1)); where ~\Theta(h-1) is computed in T;
	return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))
```



After constructing the CNTK matrix, the kernel regression problem is solved by:


```python
# H[N_train+N_test, N_train+N_test]; so H[N_train:, :N_train] = H(x_test, x_train);
u = H[N_train:, :N_train].dot(scipy.linalg.solve(H[:N_train, :N_train], Y_train))
```



##### Summary

In this post I discussed the details in NTKs for fully connected and convolutional neural networks. Hope this post would serve as a background point for possible future interests on the theoretical implications of this method. I'm very impressed with the trick that the authors employed to compute the expectation, as well as the connections drawn between NTK and Gaussian Process views of deep neural networks.




#### References

1. Arthur Jacot, Franck Gabriel, and Clément Hongler, ["Neural tangent kernel: Convergence and generalization in neural networks"](https://arxiv.org/abs/1806.07572), NeurIPS 2018;
2. Atsushi Nitanda & Taiji Suzuki, ["Optimal Rates for Averaged Stochastic Gradient Descent under Neural Tangent Kernel Regime"](https://arxiv.org/abs/2006.12297), ICLR 2021;
3. Keyulu Xu et al, ["How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks"](https://openreview.net/forum?id=UH-cmocLJC), ICLR 2021;
4. Jiri Hron et al, [Infinite attention: NNGP and NTK for deep attention networks](https://arxiv.org/abs/2006.10540), ICML 2020;
5. Sanjeev Arora el al, ["On Exact Computation with an Infinitely Wide Neural Net"](https://arxiv.org/abs/1904.11955), NeurIPS 2019;
6. Jaehoon Lee et al, ["Deep Neural Networks as Gaussian Processes"](https://arxiv.org/abs/1711.00165), ICLR 2018;

