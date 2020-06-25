---
layout: post
comments: true
title: Zoo of Deep Learning Modules
date: 2020-05-25
author: Baihua Xie
tags: foundation
mins: 20
---

> Deep Neural Networks are built with modules. This post introduces some of the most widely used differentiable modules in mainstream deep learning models, from the basic parameter-free operations such as pooling, activations, to linear, attention and the more complex reccurent modules. For each module introduced, either a set of mathematical formulations or a PyTorch/Numpy implementation of the module's forward, backward, and when applicable, parameter gradient methods is provided. Regular updates will be made to include some of the more recent progresses in literature pertaining to the design, analysis or integration of novel modules.



A neural network can be viewed as a __compute graph__, i.e., a directed acyclic graph with nodes and edges such that:

- nodes represent input, output and intermediate states, in the form of tensors 
- edges represent parameterized (e.g., affine transformation) or parameter-free (e.g., activations) functional mappings between nodes

Any compute graph is expected to perform three sets of computations:

{: class="columns is-full is-centered"}

![modular backprop](/assets/images/01_modular_backprop.png)

* __forward propagation__ (fprop), which starts from the input and propagates the mappings through the graph to produce an output and a loss; this is the process that defines the network's functional mapping and is used in inference stage
* __backward propagation__ (backprop), which starts from the loss and propagates its gradients w.r.t. the intermediate states backwards through the graph to the input; this is the process that is used in training stage to produce gradients
* __parameter gradient__, which moves through the graph along with the back propagation to produce gradients of the loss w.r.t. network parameters; these gradients are subsequently used in any gradient-based optimization methods (e.g., SGD) to update the network parameters

Modern neural networks have grown in complexity in terms of the depth, topology, mappings, etc.. Instead of treating the network's computations as a whole, people have devised a method based on the chain rule of derivatives known as __automatic differentiation__ (or auto-diff; also synonymous to the term "modular backpropagation"), which breaks the compute graph into a stack of __modules__ (a.k.a., layers). Each module is associated with three methods: __forward-pass()__, __backward-pass()__, __parameter-gradient()__. With these well-defined modules as the building blocks, the network's computations are simply produced by stacking the modules head-to-tail according to its architecture. Popular frameworks such as PyTorch have dedicated engines (autograd) to provide automatic differentiation throughout the training iterations over the compute graph. 



In general, one could devise any form of modules and incorporate these modules into a network architecture with ease, so long as the modules are differentiable and well-defined with the aforementioned methods. A good example is the insertion of Batch Normalization (BN) modules into modern CNN's; with auto-diff, this process became computationally efficient and conceptually clear. Non-differentiable modules can be trained by policy gradient methods in reinforcement learning, without the need for a backward pass, but their integration into a compute graph driven by an automatic differentiation engine could be problematic since gradients can not be naturally back-propagated through the said module. 



Below are some widely used modules:

Notation: 

* $x$: scalar
* $\vec{x}$: vectors
* $x_i$: scalar element of vector $\vec{x}$
* $X$: matrix
* vectors adopt column format
* partial derivatives adopt [numerator layout](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)

#### Parameter-free modules

| Type           | Module            | Forward-pass                                                 | Backward-pass                                                |
| :------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Element-wise   | Add               | $$\vec{y} = \vec{a} \oplus \vec{b}$$                         | $$\frac{\partial{L}}{\partial{\vec{a}}} = \frac{\partial{L}}{\partial\vec{y}}$$ |
|                | Multiply          | $$\vec{y}=\vec{a}\odot\vec{b}$$                              | $$\frac{\partial{L}}{\partial{\vec{a}}}=\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{b}$$ |
| Group          | Sum               | $$y=\sum x_i$$                                               | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{y}}\vec{1}^T $$ |
|                | Max               | $$y=max\{x_i\}$$                                             | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y}}&\;\;\text{if }i\text{ was maximal}\\0&\;\;\text{if otherwise}\end{cases}$$ |
|                | Switch            | $$\vec{y}=\vec{s}\odot\vec{x}$$                              | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{s}^T$$ |
| Activation     | Relu              | $$y_i=max\{0,x_i\}$$                                         | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{42pt}\text{if }x_i>0\\0&\hspace{42pt}\text{if otherwise}\end{cases}$$ |
|                | Leaky Relu        | $$y_i=max\{0.01x_i,x_i\}$$                                   | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{24pt}\text{if }x_i>0\\0.01\frac{\partial{L}}{\partial{y_i}}&\hspace{24pt}\text{if otherwise}\end{cases}$$ |
|                | Elu               | $$y_i=\begin{cases}x_i&\;\;\text{if }x_i>0\\\alpha(e^{x_i}-1)&\;\;\text{if otherwise}\end{cases}$$ | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{7pt}\text{if }x_i>0\\\frac{\partial{L}}{\partial{y_i}}(y_i+\alpha)&\hspace{7pt}\text{if otherwise}\end{cases}$$ |
|                | Selu              | $$y_i=\begin{cases}\lambda x_i&\text{if }x_i>0\\\lambda\alpha(e^{x_i}-1)&\text{if otherwise}\end{cases}$$ | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}\lambda&\hspace{1pt}\text{if }x_i>0\\\frac{\partial{L}}{\partial{y_i}}(y_i+\lambda\alpha)&\hspace{1pt}\text{if otherwise}\end{cases}$$ |
|                | Sigmoid           | $$y_i=\frac{1}{1+e^{-x_i}}$$                                 | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(\vec{y}(\vec{1}-\vec{y})^T)$$ |
|                | Softmax           | $$y_i=\frac{e^{x_i}}{\sum\limits_{\forall k}e^{x_k}}$$       | $$\frac{\partial{L}}{\partial{\vec{x}}}=\vec{s}^T-\vec{y}^T\sum\limits_{\forall{i}}s_i\\\text{ where }s_i=\frac{\partial{L}}{\partial{y_i}}y_i$$ |
|                | Softplus          | $$y_i=\log(e^{x_i}+1)$$                                      | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(\frac{e^{x_i}}{e^{x_i}+1})$$ |
|                | Tanh              | $$y_i=\frac{e^{x_i}-e^{-x_i}}{e^{x_i}+e^{-x_i}}$$            | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(I-\vec{y}\vec{y}^T)$$ |
| Loss           | Cross-Entropy     | $$L=-\sum\limits_{\forall{i}}p_i\log{x_i}$$                  | $$\frac{\partial{L}}{\partial{x_i}}=-\frac{p_i}{x_i}$$       |
|                | Squared Error     | $$L=\lVert{\vec{t}-\vec{x}}\rVert^2=\sum\limits_{\forall{i}}(t_i-x_i)^2$$ | $$\frac{\partial{L}}{\partial{\vec{x}}}=-2(\vec{t}-\vec{x})^T$$ |
| Pooling        | MaxPooling-2D     | $$\begin{split}y_{i,j}&=max(x_{Ui:Ui+p,Uj:Uj+q})\\\text{U: }&\text{stride}\\\end{split}$$ | $$\DeclareMathOperator*{\argmax}{argmax}\frac{\partial{L}}{\partial{x_{i,j}}}=\begin{cases}\sum\limits_{i'}^H\sum\limits_{j'}^H\frac{\partial{L}}{\partial{y_{i',j'}}}&\hspace{10pt}\text{if }\argmax\limits_{i',j'}\{x_{Ui':Ui'+p,Uj':Uj'+p}\}=i,j\\0&\hspace{10pt}\text{otherwise}\end{cases}$$ |
|                | AveragePooling-2D | $$y_{i,j}=\frac{1}{pq}\sum\limits_p^P\sum\limits_q^Qx_{Ui+p,Uj+q}$$ | $$\frac{\partial{L}}{\partial{x_{i,j}}}=\frac{1}{pq}\sum\limits_{\frac{i-p}{U}}^P\sum\limits_{\frac{j-q}{U}}^Q\frac{\partial{L}}{\partial{y_{\frac{i-p}{U},\frac{j-q}{U}}}}$$ |
| Regularization | Drop-out          | $$\vec{y}=\frac{1}{1-q}\vec{D}\odot\vec{x}\\\text{where }\vec{D}\text{ is the drop-out swith vector}$$ | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{1}{1-q}\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{D}^T$$ |

#### Linear modules

##### FC

* __forward-pass__

$$
\vec{y}=W\vec{x}+\vec{b}
$$

* __backward-pass__

$$
\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}W
$$

* __parameter-pass__

$$
\frac{\partial{L}}{\partial{W}}=\frac{\partial{L}}{\partial{\vec{y}}}(I\otimes\vec{x})=\vec{x}\frac{\partial{L}}{\partial{\vec{y}}}\\\otimes\text{:kronecker product}\\\frac{\partial{L}}{\partial{\vec{b}}}=\frac{\partial{L}}{\partial{\vec{y}}}
$$

##### Conv-1D

* __forward-pass__

$$
\begin{split}y_{s,i}&=x_{p,i}*w_{s,p,i'}=\sum\limits_{p}^C\sum\limits_{i'}^Kx_{p,i-i'}\cdot{w_{s,p,i'}}\\s,p&\text{: output/input channel index}\\i,j,k&\text{: activation index}\\K&\text{: kernel size}\\H&\text{: fmap size}\\C&\text{: input channels}\\M&\text{: output channels}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i}}}&=\sum\limits_{s}^Mw_{s,p,i}^T*\frac{\partial{L}}{\partial{y_{s,i'}}}=\sum\limits_{s}^M\sum\limits_{i'}^Hw_{s,p,i-i'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i'}}}}\end{split}\\w^T\text{: rotate }180^o
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{s,p,i}}}&=x_{p,i}^T*\frac{\partial{L}}{\partial{y_{s,i'}}}=\sum\limits_{i'}^Hx_{p,i-i'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i'}}}}\end{split}\\x^T\text{: rotate }180^o
$$

##### Conv-2D

* __forward-pass__

$$
\begin{split}y_{s,i,j}&=x_{p,i,j}*w_{s,p,i',j'}=\sum\limits_{p}^C\sum\limits_{i'}^K\sum\limits_{j'}^Kx_{p,i-i',j-j'}\cdot{w_{s,p,i',j'}}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i,j}}}&=\sum\limits_{s}^Mw_{s,p,i,j}^T*\frac{\partial{L}}{\partial{y_{s,i',j'}}}=\sum\limits_{s}^M\sum\limits_{i'}^H\sum\limits_{j'}^Hw_{s,p,i-i',j-j'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{s,p,i,j}}}&=x_{p,i,j}^T*\frac{\partial{L}}{\partial{y_{s,i',j'}}}=\sum\limits_{i'}^H\sum\limits_{j'}^Hx_{p,i-i',j-j'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\end{split}
$$

##### Conv-3D

* __forward-pass__

$$
\begin{split}y_{s,i,j,k}&=x_{p,i,j,k}*w_{s,p,i',j',k'}=\sum\limits_{p}^C\sum\limits_{i'}^K\sum\limits_{j'}^K\sum\limits_{k'}^Kx_{p,i-i',j-j',k-k'}\cdot{w_{s,p,i',j',k'}}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i,j,k}}}&=\sum\limits_{s}^Mw_{s,p,i,j,k}^T*\frac{\partial{L}}{\partial{y_{s,i',j',k'}}}=\sum\limits_{s}^M\sum\limits_{i'}^H\sum\limits_{j'}^H\sum\limits_{k'}^Hw_{s,p,i-i',j-j',k-k'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j',k'}}}}\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{s,p,i,j,k}}}&=x_{p,i,j,k}^T*\frac{\partial{L}}{\partial{y_{s,i',j',k'}}}=\sum\limits_{i'}^H\sum\limits_{j'}^H\sum\limits_{k'}^Hx_{p,i-i',j-j',k-k'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j',k'}}}}\end{split}
$$

##### Separable-Conv-2D

* __forward-pass__

$$
\begin{split}o_{p,i,j}&=x_{p,i,j}*w_{p,i',j'}^1=\sum\limits_{i'}^K\sum\limits_{j'}^Kx_{p,i-i',j-j'}\cdot{w_{p,i',j'}^1}\\y_{s,i,j}&=o_{p,i,j}*w_{s,p}^2=\sum\limits_{p}^Co_{p,i,j}\cdot{w_{s,p}^2}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i,j}}}&=\sum\limits_{s}^Mw_{s,p}^2(w_{p,i,j}^1*\frac{\partial{L}}{\partial{y_{s,i',j'}}})=\sum\limits_{s}^Mw_{s,p}^2\sum\limits_{i'}^H\sum\limits_{j'}^Hw_{s,p,i-i',j-j'}^{1,T}\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{p,i,j}^1}}&=\sum\limits_{s}^Mw_{s,p}^2\sum\limits_{i'}^H\sum\limits_{j'}^Hx_{p,i-i',j-j'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\\\frac{\partial{L}}{\partial{w_{s,p}^2}}&=\sum\limits_{i}^H\sum\limits_{j}^H\frac{\partial{L}}{\partial{y_{s,i,j}}}\cdot{o_{p,i,j}}\end{split}
$$

#### Normalization

##### BN

* __forward-pass__

$$
\begin{split}&y_m=\gamma\frac{x_m-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta\\&\mu_B=\frac{1}{M}\sum\limits_mx_m\\&\sigma_B^2=\frac{1}{M}\sum\limits_m(x_m-\mu_B)^2\\&x_m\text{: }\text{i-th element in m-th sample's vector}\\&\text{(omit subscript i for simplicity)}\\&\text{M: }\text{batch size}\\\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{\sigma_B^2}}&=\sum\limits_m\frac{\partial{L}}{\partial{y_m}}\gamma(x_m-\mu_B)(-\frac{1}{2})(\sigma_B^2+\epsilon)^{-\frac{3}{2}} \\\frac{\partial{L}}{\partial{\mu_B}}&=\sum\limits_m\frac{\partial{L}}{\partial{y_m}}\gamma(-\frac{1}{\sqrt{\sigma_B^2+\epsilon}})+\frac{\partial{L}}{\partial{\sigma_B^2}}\frac{-\sum\limits_m2(x_m-\mu_B)}{m}\\\frac{\partial{L}}{\partial{x_m}}&=\frac{\partial{L}}{\partial{y_m}}\gamma\frac{1}{\sqrt{\sigma_B^2+\epsilon}}+\frac{\partial{L}}{\partial{\mu_B}}\frac{1}{m}+\frac{\partial{L}}{\partial{\sigma_B^2}}\frac{2}{m}(x_m-\mu_B)\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{\gamma}}&=\sum\limits_m\frac{\partial{L}}{\partial{y_m}}\frac{x_m-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\\\frac{\partial{L}}{\partial{\beta}}&=\sum\limits_m\frac{\partial{L}}{\partial{y_m}}\end{split}
$$

#### Reccurent modules

##### SimpleRNN

* __forward-pass__

$$
\begin{split}\vec{h}_{t+1}&=tanh(W\vec{h}_t+U\vec{x}_t)\\\vec{o}_t&=V\vec{h}_t\end{split}
$$

* __backward-pass__

$$
\begin{split}&\frac{\partial{L}}{\partial{\vec{h}_t}}=\frac{\partial{L}}{\partial{\vec{o}_t}}V+\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}diag(I-\vec{h}_{t+1}\vec{h}_{t+1}^T)W\\&\frac{\partial{L}}{\partial{\vec{h}_\tau}}=\frac{\partial{L}}{\partial{\vec{o}_\tau}}V\\(&\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\;\text{obtained by BPTT from last time step }\tau)\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{W}}&=\sum\limits_t\vec{h}_t\frac{\partial{L}}{\partial{\vec{h}_t}}diag(I-\vec{h}_t\vec{h}_t^T)\\\frac{\partial{L}}{\partial{U}}&=\sum\limits_t\vec{x}_t\frac{\partial{L}}{\partial{\vec{h}_t}}diag(I-\vec{h}_t\vec{h}_t^T)\\\frac{\partial{L}}{\partial{V}}&=\sum\limits_t\vec{x}_t\frac{\partial{L}}{\partial{\vec{o}_t}}\end{split}
$$

##### LSTM

* __forward-pass__

$$
\begin{split}\text{cell state: }&\vec{C}_{t+1}=\vec{f}_t\odot\vec{C}_t+\vec{i}_t\odot\vec{G}_t\\\text{hidden state: }&\vec{h}_{t+1}=\vec{o}_t\odot{tanh(\vec{C}_{t+1})}\\\\\text{forget gate: }&\vec{f}_t=\sigma(W_f\vec{h}_t+U_f\vec{x}_t)\\\text{input gate: }&\vec{i}_t=\sigma(W_i\vec{h}_t+U_i\vec{x}_t)\\\text{output gate: }&\vec{o}_t=\sigma(W_o\vec{h}_t+U_o\vec{x}_t)\\\text{cell gate: }&\vec{G}_t=tanh(W_g\vec{h}_t+U_g\vec{x}_t)\\\sigma()\text{: }&\text{sigmoid function}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{\vec{h}_t}}&=\frac{\partial{L}}{\partial{\vec{G}_t}}diag(I-\vec{G}_t\vec{G}_t^T)W_g+\frac{\partial{L}}{\partial{\vec{i}_t}}diag(\vec{i}_t(\vec{1}-\vec{i}_t)^T)W_i+\frac{\partial{L}}{\partial{\vec{f}_t}}diag(\vec{f}_t(\vec{1}-\vec{f}_t)^T)W_f+\frac{\partial{L}}{\partial{\vec{o}_t}}diag(\vec{o}_t(\vec{1}-\vec{o}_t)^T)W_o+\frac{\partial{L_t}}{\partial{\vec{h}_t}}\\
\frac{\partial{L}}{\partial{\vec{x}_t}}&=\frac{\partial{L}}{\partial{\vec{G}_t}}diag(I-\vec{G}_t\vec{G}_t^T)U_g+\frac{\partial{L}}{\partial{\vec{i}_t}}diag(\vec{i}_t(\vec{1}-\vec{i}_t)^T)U_i+\frac{\partial{L}}{\partial{\vec{f}_t}}diag(\vec{f}_t(\vec{1}-\vec{f}_t)^T)U_f+\frac{\partial{L}}{\partial{\vec{o}_t}}diag(\vec{o}_t(\vec{1}-\vec{o}_t)^T)U_o\\\frac{\partial{L}}{\partial{\vec{C}_t}}&=\frac{\partial{L}}{\partial{\vec{C}_{t+1}}}\odot\vec{f}_t^T+\frac{\partial{L}}{\partial{\vec{h}_t}}\odot(\vec{o}_t^Tdiag(I-tanh(\vec{C}_{t+1})tanh(\vec{C}_{t+1}^T)))\\\\\frac{\partial{L}}{\partial{\vec{G}_t}}&=\frac{\partial{L}}{\partial{\vec{C}_{t+1}}}\odot\vec{i}_t\hspace{20pt}\frac{\partial{L}}{\partial{\vec{f}_t}}=\frac{\partial{L}}{\partial{\vec{C}_{t+1}}}\odot\vec{C}_t\hspace{20pt}\frac{\partial{L}}{\partial{\vec{i}_t}}=\frac{\partial{L}}{\partial{\vec{C}_{t+1}}}\odot\vec{G}_t\hspace{20pt}\frac{\partial{L}}{\partial{\vec{o}_t}}=\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\odot{tanh(\vec{C}_{t+1})}\\&(\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\text{ & }\frac{\partial{L}}{\partial{\vec{C}_{t+1}}}\text{obtained by BPTT)}\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{W_g}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{G}_t}}diag(I-\vec{G}_t\vec{G}_t^T)\hspace{20pt}&\frac{\partial{L}}{\partial{U_g}}=\vec{x}_t\frac{\partial{L}}{\partial{\vec{G}_t}}diag(I-\vec{G}_t\vec{G}_t^T)\\\frac{\partial{L}}{\partial{W_f}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{f}_t}}diag(\vec{f}_t(\vec{1}-\vec{f}_t)^T)\hspace{20pt}&\frac{\partial{L}}{\partial{U_f}}=\vec{x}_t\frac{\partial{L}}{\partial{\vec{f}_t}}diag(\vec{f}_t(\vec{1}-\vec{f}_t)^T)\\\frac{\partial{L}}{\partial{W_i}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{i}_t}}diag(\vec{i}_t(\vec{1}-\vec{i}_t)^T)\hspace{20pt}&\frac{\partial{L}}{\partial{U_i}}=\vec{x}_t\frac{\partial{L}}{\partial{\vec{i}_t}}diag(\vec{i}_t(\vec{1}-\vec{i}_t)^T)\\\frac{\partial{L}}{\partial{W_o}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{o}_t}}diag(\vec{o}_t(\vec{1}-\vec{o}_t)^T)\hspace{20pt}&\frac{\partial{L}}{\partial{U_o}}=\vec{x}_t\frac{\partial{L}}{\partial{\vec{o}_t}}diag(\vec{o}_t(\vec{1}-\vec{o}_t)^T)\\\end{split}
$$

##### GRU

* __forward-pass__

$$
\begin{split}\text{hidden staet: }&\vec{h}_{t+1}=(\vec{1}-\vec{z}_t)\odot\vec{h}_t+\vec{z}_t\odot\vec{c}_t\\\\ \text{cell state: }&\vec{c}_t=tanh(W_c\vec{r}_t\odot\vec{h}_t+U_c\vec{x}_t)\\\text{update gate: }&\vec{z}_t=\sigma(W_z\vec{h}_t+U_z\vec{x}_t)\\\text{reset gate: }&\vec{r}_t=\sigma(W_r\vec{h}_t+U_r\vec{x}_t)\\\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{\vec{h}_t}}&=\frac{\partial{L}}{\partial{\vec{z}_t}}diag(\vec{z}_t(\vec{1}-\vec{z}_t)^T)W_z+\frac{\partial{L}}{\partial{\vec{r}_t}}diag(\vec{r}_t(\vec{1}-\vec{r}_t)^T)W_r+\frac{\partial{L}}{\partial{\vec{c}_t}}diag(I-\vec{c}_t\vec{c}_t^T)W_c\odot\vec{r}_t^T+\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\odot(\vec{1}-\vec{z}_t)^T+\frac{\partial{L_t}}{\partial{\vec{h}}_t}\\\frac{\partial{L}}{\partial{\vec{x}_t}}&=\frac{\partial{L}}{\partial{\vec{z}_t}}diag(\vec{z}_t(\vec{1}-\vec{z}_t)^T)U_z+\frac{\partial{L}}{\partial{\vec{r}_t}}diag(\vec{r}_t(\vec{1}-\vec{r}_t)^T)U_r+\frac{\partial{L}}{\partial{\vec{c}_t}}diag(I-\vec{c}_t\vec{c}_t^T)U_c\\\frac{\partial{L}}{\partial{\vec{z}_t}}&=\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\odot{(\vec{c}_t-\vec{h}_t)^T\hspace{35pt}}\frac{\partial{L}}{\partial{\vec{c}_t}}=\frac{\partial{L}}{\partial{\vec{h}_{t+1}}}\odot\vec{z}_t\hspace{35pt}\frac{\partial{L}}{\partial{\vec{r}_t}}=\frac{\partial{L}}{\partial{\vec{c}_t}}diag(I-\vec{c}_t\vec{c}_t^T)Wc\odot\vec{h}_t^T\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{W_z}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{z}_t}}diag(\vec{z}_t(\vec{1}-\vec{z}_t)^T)\hspace{35pt}\frac{\partial{L}}{\partial{W_r}}&=\vec{h}_t\frac{\partial{L}}{\partial{\vec{r}_t}}diag(\vec{r}_t(\vec{1}-\vec{r}_t)^T)\hspace{35pt}\frac{\partial{L}}{\partial{W_c}}&=(\vec{r}_t\odot\vec{h}_t)\frac{\partial{L}}{\partial{\vec{c}_t}}diag(I-\vec{c}_t\vec{c}_t^T)\\\frac{\partial{L}}{\partial{U_z}}&=\vec{x}_t\frac{\partial{L}}{\partial{\vec{z}_t}}diag(\vec{z}_t(\vec{1}-\vec{z}_t)^T)\hspace{35pt}\frac{\partial{L}}{\partial{U_r}}&=\vec{x}_t\frac{\partial{L}}{\partial{\vec{r}_t}}diag(\vec{r}_t(\vec{1}-\vec{r}_t)^T)\hspace{35pt}\frac{\partial{L}}{\partial{U_c}}&=\vec{x}_t\frac{\partial{L}}{\partial{\vec{c}_t}}diag(I-\vec{c}_t\vec{c}_t^T)\\\end{split}
$$







## References

1. Simon Osindero, "Lecture-03: Neural Network Foundations" , the DeepMindxUCL lectures
2. Ian Goodfellow, Yoshua Bengio, Aaron Courvill, "Deep Learning", Chap. 10 (RNN, LSTM, GRU)
3. Jefkine, ["Backpropagation In Convolutional Neural Networks"](https://canvas.stanford.edu/files/1041875/download?download_frd=1&verifier=tFv4Jc7bCezxJg9rG2yhEKEERi70zJ3ScmFbNlbN) 
4. Christopher Olah, ["Understanding LSTM networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
5. Sergey Ioffe, Christian Szegedy, ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167)
6. Van Huyen DO, ["Back propagation in Long Short Term Memory (LSTM)"](https://medium.com/@dovanhuyen2018/back-propagation-in-long-short-term-memory-lstm-a13ad8ae7a57)