---
layout: post
comments: true
title: Convolutional Neural Tangent Kernels
date: 2021-05-08
author: Baihua Xie
tags: neural-tangent-kernels theory
mins: 20-min

---

Since proposed by Jacob el al<sup>1</sup> in 2018, _neural tangent kernels_ have become an important tool in recent literature<sup>2,3,4</sup> that study the theoretical behaviors of deep neural networks. In this post I'll discuss one of the earlier works on convolutional neural tangent kernels, or CNTK, by Arora et al^5^ in NeurIPS 2019. In this paper, the authors derived an efficient algorithm that computes a CNTK kernel regression problem which the authors claim could estimate the training dynamics of ReLU CNN's trained with gradient descent. The CNTK algorithm achieves 77.4% accuracy on the CIFAR-10 dataset. Although a poor performance compared to well-trained deep CNN's, this is nonetheless an impressive +10% higher than previous state-of-the-art for a purely kernel-based method. Since the main significance of NTKs lies with them being a powerful tool to study the dynamics of training deep neural networks, one could argue that the absolute accuracy numbers are of less concern. In this post I'll focus on how the CNTK works exactly, as well as what kind of assumptions and limitations it might have for studying mainstream deep neural networks.

First let's get a few concepts out of the way:

**Kernel Regression.** Let $$f(\theta;x)\in\mathbb{R}$$ denote the output of a neural network given weights $$\theta$$ and an input sample $$x$$. Let $$X$$ and $$X^*$$ represent the set of training and testing dataset, respectively. A regression or classification problem would ask the network to predict $$f(\theta; x^*)$$ given the training samples $$\{X,y\}$$. Kernel regression solves this problem by giving a closed-form expression for the prediction as the following:
$$
f(\theta;x^*)=\sum_j^N\mathcal{k}(x^*,x_j)y_j
$$

Here, $$\{x_j,y_j\}\in\{X,y\}$$ are the $N$ training samples; $$\mathcal{k}(\cdot,\cdot)$$ is a kernel function between two samples that produces a similarity score. In essence, kernel regression produces a prediction for an unseen sample in the test set as a weighted combination of the labels in the training set.

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

In summary, by assuming the network's width can go to inftyity, the NTK formulation proposed that, instead of having to train the neural network using gradient descent and backpropagation, one can simply compute the predicted outputs in the following steps:

1. Initialize the weights;
2. Compute the NTK matrix $$H'$$ for all training samples;
3. With each new test sample $$x^*$$, compute the NTK kernel score of $$\mathcal{k}(x^*,x_i)$$ w.r.t. each sample in the training set;
4. Use the closed-form solution to compute the predicted output;

At a glance, this formulation goes against the fundamental ideas behind representation learning, which is what gives the expressive powers to deep neural networks. One can observe that this procedure has several serious problems:

1. The weights are only initialized, never updated; if this is the case then why do we need the weights to begin with? also where does the expectation w.r.t. initialization distribution go during the above derivation of NTKs? 
2. For every new test sample we need to re-evaluate the solution on the entire training set; this can have huge computational complexity implications, as the training set in practice is usually very large;
3. Can this solution overfit the training set to begin with? i.e., if I pass a training sample to the formula, is it guaranteed to output the ground truth?

I'll discuss the insights and limitations derived from the NTK formulation in a later post. For now let's turn to the main problem being addressed in the Arora paper. Since the NTK is not a closed-form function, approximations would be required to use it in practical networks. Here I'll follow the derivation of an approximated NTK for networks comprised of only fully connected (or feed-forward) layers with ReLU activations (because it's relatively easier to work out...). The NTK for CNN's follows very similar derivations, as CNN's are essentially fully connected networks sparsified through weight sharing.

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
\frac{\partial f^{(h+1)}(x)}{\partial f^{(h)}(x)}=\text{diag}(\sigma(f^{(h)}(x)))(W^{(h+1)})^T
$$
(there is a missing derivative symbol) Let's check the dimensions here: $$f^{(h+1)}\in\mathbb{R}^{d_{h+1}}$$ and $$f^{(h)}\in\mathbb{R}^{d_{h}}$$ are column vectors; if we consider using the denominator layout, the L.H.S. should be a matrix $$\in\mathbb{R}^{d_{h}\times d_{h+1}}$$; on the R.H.S., the diagonal matrix is $$\in\mathbb{R}^{d_{h}\times d_{h}}$$ and the transposed weight matrix is $$\in\mathbb{R}^{d_{h}\times d_{h+1}}$$; the dimensions match under denominator layout.

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





#### References

1. Arthur Jacot, Franck Gabriel, and Clément Hongler, "Neural tangent kernel: Convergence and generalization in neural networks", 2018
2. 

