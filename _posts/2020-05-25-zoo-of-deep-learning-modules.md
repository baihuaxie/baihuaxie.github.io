---
layout: post
comments: true
title: Zoo of Deep Learning Modules
date: 2020-05-25
author: Baihua Xie
tags: foundation
mins: long
---

> Modules are the building blocks of deep learning. This post introduces some of the most widely used differentiable modules in modern networks, from the basic parameter-free operations such as pooling, activations, to linear, attention and the more complex reccurent modules. For each module introduced, either a set of mathematical formulations or a PyTorch/Numpy implementation of the module's forward, backward, and when applicable, parameter gradient methods is provided. Regular updates will be made to include some of the more recent progresses in literature pertaining to the design, analysis or integration of novel modules.

Deep learning is a heavily loaded term. Sometimes it is used to refer to concepts more suitably described by terms such as deep neural networks, supervised learning, representation learning, etc. Yan LeCun gave a generalized definition of deep learning at a keynote event in AAAI 2020 as follows:

"Deep learning is building a system by assembling parameterized __modules__ into a (possibly dynamic) __computation graph__, and training it to perform a task by __optimizing__ the parameters using a __gradient-based method__."

This blog post will provide a glossary of the modules, which are essentially the building blocks of modern deep learning systems. Note that only differentiable modules would be included here, as they could be trained by gradient-based methods in a network.

Any differentiable module is expected to perform three sets of computations:

{: class="columns is-full is-centered"}

![modular backprop](/assets/images/01_modular_backprop.png)

* __forward propagation__ (fprop), which starts from the input and propagates the mappings through the graph to produce an output and a loss; this is the process that defines the network's functional mapping and is used in inference stage
* __backward propagation__ (backprop), which starts from the loss and propagates its gradients w.r.t. the intermediate states backwards through the graph to the input; this is the process that is used in training stage to produce gradients
* __parameter gradient__, which moves through the graph along with the back propagation to produce gradients of the loss w.r.t. network parameters; these gradients are subsequently used in any gradient-based optimization methods (e.g., SGD) to update the network parameters

Karen Osidero's [lecture at UCL](https://www.youtube.com/watch?v=5eAXoPSBgnE&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs&index=3) provided a good reference as to why these three methods are needed to define a module. 

In simple terms, modern frameworks such as TensorFlow or PyTorch are based on automatic differentiation, which works by tracking the gradients' propagation through the network as a stack of well-defined, differentiable modules. A designer need not worry about the cumbersome details of gradient calculations, but are only tasked with specifying the modules and their input-output relationships. In general, one could devise any form of modules and incorporate these modules into a network architecture with ease, so long as the modules are differentiable and are defined with the aforementioned three methods.

Below are some widely used modules:

Notation: 

* $$x$$: scalar
* $$\vec{x}$$: vectors
* $$x_i$$: scalar element of vector $$\vec{x}$$
* $$X$$: matrix
* vectors adopt column format; partial derivatives adopt the [numerator layout](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)

#### Parameter-free modules

Functional layers such as element-wise operations, activations, loss functions, etc., although parameter-free, are still most often implemented following the same modular definition. This facilitates the insertion of the functional layers into a network definition with ease.

| Type           | Module                 | Forward-pass                                                 | Backward-pass                                                |
| :------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Element-wise   | Add                    | $$\vec{y} = \vec{a} \oplus \vec{b}$$                         | $$\frac{\partial{L}}{\partial{\vec{a}}} = \frac{\partial{L}}{\partial\vec{y}}$$ |
|                | Multiply               | $$\vec{y}=\vec{a}\odot\vec{b}$$                              | $$\frac{\partial{L}}{\partial{\vec{a}}}=\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{b}$$ |
| Group          | Sum                    | $$y=\sum x_i$$                                               | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{y}}\vec{1}^T $$ |
|                | Max                    | $$y=max\{x_i\}$$                                             | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y}}&\;\;\text{if }i\text{ was maximal}\\0&\;\;\text{if otherwise}\end{cases}$$ |
|                | Switch                 | $$\vec{y}=\vec{s}\odot\vec{x}$$                              | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{s}^T$$ |
| Activation     | Relu<sup>1</sup>       | $$y_i=max\{0,x_i\}$$                                         | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{42pt}\text{if }x_i>0\\0&\hspace{42pt}\text{if otherwise}\end{cases}$$ |
|                | Leaky Relu<sup>2</sup> | $$y_i=max\{0.01x_i,x_i\}$$                                   | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{24pt}\text{if }x_i>0\\0.01\frac{\partial{L}}{\partial{y_i}}&\hspace{24pt}\text{if otherwise}\end{cases}$$ |
|                | Elu<sup>3</sup>        | $$y_i=\begin{cases}x_i&\;\;\text{if }x_i>0\\\alpha(e^{x_i}-1)&\;\;\text{if otherwise}\end{cases}$$ | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}&\hspace{7pt}\text{if }x_i>0\\\frac{\partial{L}}{\partial{y_i}}(y_i+\alpha)&\hspace{7pt}\text{if otherwise}\end{cases}$$ |
|                | Selu<sup>4</sup>       | $$y_i=\begin{cases}\lambda x_i&\text{if }x_i>0\\\lambda\alpha(e^{x_i}-1)&\text{if otherwise}\end{cases}$$ | $$\frac{\partial{L}}{\partial{x_i}}=\begin{cases}\frac{\partial{L}}{\partial{y_i}}\lambda&\hspace{1pt}\text{if }x_i>0\\\frac{\partial{L}}{\partial{y_i}}(y_i+\lambda\alpha)&\hspace{1pt}\text{if otherwise}\end{cases}$$ |
|                | Sigmoid                | $$y_i=\frac{1}{1+e^{-x_i}}$$                                 | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(\vec{y}(\vec{1}-\vec{y})^T)$$ |
|                | Softmax                | $$y_i=\frac{e^{x_i}}{\sum\limits_{\forall k}e^{x_k}}$$       | $$\frac{\partial{L}}{\partial{\vec{x}}}=\vec{s}^T-\vec{y}^T\sum\limits_{\forall{i}}s_i\\\text{ where }s_i=\frac{\partial{L}}{\partial{y_i}}y_i$$ |
|                | Softplus<sup>1</sup>   | $$y_i=\log(e^{x_i}+1)$$                                      | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(\frac{e^{x_i}}{e^{x_i}+1})$$ |
|                | Tanh                   | $$y_i=\frac{e^{x_i}-e^{-x_i}}{e^{x_i}+e^{-x_i}}$$            | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{\partial{L}}{\partial{\vec{y}}}diag(I-\vec{y}\vec{y}^T)$$ |
| Loss           | Cross-Entropy          | $$L=-\sum\limits_{\forall{i}}p_i\log{x_i}$$                  | $$\frac{\partial{L}}{\partial{x_i}}=-\frac{p_i}{x_i}$$       |
|                | Squared Error          | $$L=\lVert{\vec{t}-\vec{x}}\rVert^2=\sum\limits_{\forall{i}}(t_i-x_i)^2$$ | $$\frac{\partial{L}}{\partial{\vec{x}}}=-2(\vec{t}-\vec{x})^T$$ |
| Pooling        | MaxPooling2D           | $$\begin{split}y_{i,j}&=max(x_{Ui:Ui+p,Uj:Uj+q})\\\text{U: }&\text{stride}\\\end{split}$$ | $$\DeclareMathOperator*{\argmax}{argmax}\frac{\partial{L}}{\partial{x_{i,j}}}=\begin{cases}\sum\limits_{i'}^H\sum\limits_{j'}^H\frac{\partial{L}}{\partial{y_{i',j'}}}&\hspace{10pt}\text{if }\argmax\limits_{i',j'}\{x_{Ui':Ui'+p,Uj':Uj'+p}\}=i,j\\0&\hspace{10pt}\text{otherwise}\end{cases}$$ |
|                | AveragePooling2D       | $$y_{i,j}=\frac{1}{pq}\sum\limits_p^P\sum\limits_q^Qx_{Ui+p,Uj+q}$$ | $$\frac{\partial{L}}{\partial{x_{i,j}}}=\frac{1}{pq}\sum\limits_{\frac{i-p}{U}}^P\sum\limits_{\frac{j-q}{U}}^Q\frac{\partial{L}}{\partial{y_{\frac{i-p}{U},\frac{j-q}{U}}}}$$ |
| Regularization | Drop-out<sup>7</sup>   | $$\vec{y}=\frac{1}{1-q}\vec{D}\odot\vec{x}\\\text{where }\vec{D}\text{ is the drop-out swith vector}$$ | $$\frac{\partial{L}}{\partial{\vec{x}}}=\frac{1}{1-q}\frac{\partial{L}}{\partial{\vec{y}}}\odot\vec{D}^T$$ |

#### Linear modules

Linear layers are used to perform affine transformations. Usually they are the layers with the majority of learnable parameters in a network, thus serving as the cornerstones. Commonly used linear layers include fully-connected layers and convolutional layers.

##### FC

Fully connected layer (FC for short) can be traced back to the early days of artificial neural networks, where they are used to construct the multi-layer perceptron<sup>8</sup> model. For a considerable period of time (1960's - early 2000's) the fully connected layers are the only major building blocks for neural networks. In modern models, however, FC layers are no longer used throughout the network, as they consume too much computation budgets for the data-intensive training today. Instead, FC layers typically function as:

* the projection or embedding layers in neural language models using pretrained word embeddings
* a last-level projection layer to project the features into class logits in convolutional networks
* some network-in-network setting, e.g., used to for the squeeze and excitation operations in networks with SE optimizations<sup>10</sup>

Below is the modular formulation of an FC layer. Note that parameter-pass involves a vector-by-matrix derivative, which could be represented compactly by the [kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) with an identity matrix. In practice this simply reduces to a vector multiplication by input vector $$x$$ and producing a matrix. 

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

##### Conv-2D

Convolutional layers could be viewed as a FC layer with weight sharing to reduce computation<sup>9</sup>. As convolutional layers are most commonly used to process two-dimensional image data, conv-2d is first introduced, followed by the extensions to 1d (e.g., sequential text data) and 3d (video data) cases.

Convolutional layers have been firmly established as the cornerstones to modern computer vision networks. Dan C.Ciresan et al<sup>10</sup>  first reported the use of GPU implementation to train a network built with convolutional filters and max-pooling layers. AlexNet<sup>11</sup> was the work that propelled convolutional neural networks into the state-of-the-art model for image classificiation, and later many other computer vision tasks, by winning the ILSVRC 2012 challenge with a 57.2% top-5 accuracy. This work has popularized several key enabling techniques, including ReLU, dropout, grouped convolution, and distributed GPU training, as well as raising awareness to the important topics of normalization and initialization in modern networks.

After AlexNet, the design of convolutional networks (for feature extraction, or the 'back-bone' network as commonly referred to in recent literature) could be roughly divided into three development stages, as categorized by the driving factors for the performance gains and as signified by two seminar works, the ResNet<sup>12</sup> in 2015 and NAS<sup>13</sup> in 2017: 

* __unit-level design__: before ResNet, research had been focused on the the building blocks of a convolutional neural network. Major works include the Inception family, which popularized the use of factorized 3x3/1x1 convolutional filters<sup>14</sup> and batch normalization<sup>15</sup> layers. Although Xception<sup>16</sup> came after ResNet and used skip connection, its primary contribution was that it pushed the factorization into extreme by stacking a network with purely depthwise separable filters. These works went beyond the simple yet powerful idea of increasing network depth<sup>17</sup> to gain performance, instead they attempted to make the convolutional modules more expressive or computationally more efficient.
* __block-level design__: while the Inception papers already started to use complex cells with multiple parallel data paths as the building block, the ResNet-like multi-layer bottleneck block became the standard choice in recent state-of-the-art networks. Along with its influential skip connection, which is now a standard way to design convolutional network, ResNet had also popularized the use of bottleneck block, consisting of a 1x1-conv / 3x3-conv / 1x1-conv stack, to reduce computation complexity by forcing a channel-wise bottleneck before the expensive 3x3 convolutional filters<sup>12</sup>. This design pattern has been favored by several major works later, partly due to its flexibility, as a network became increasing aware of its resource budget. ResNet variants such as ResNeXt<sup>18</sup> and WideResNet<sup>19</sup> are all modifications or extensions based on the bottleneck block. MobileNet-V1<sup>20</sup> further reduced computation by using a 3x3 depthwise filter. Its successor, MobileNet-V2<sup>21</sup>, proposed an inverted and linearized version of the bottleneck block, where the expansion factor is greater than 1 and the last-layer activation is removed. The ShuffleNets<sup>22, 23</sup> extended the concept of grouped filtering firstly into the pointwise kernels by introducing a channel shuffling operation, and later direclty into the block-level processing by introducing a channel splitting operation. Other notable block-level innovations include DenseNet<sup>24</sup>, where the idea of skip connection is extended from connecting consecutive layers to connecting all pairs of layers in a dense block.
* __network-level design__: researchers have always agreed that scaling up a network could most likely improve its performance, albeit with diminishing gains. What they have not agreed on is how to best scale a network. There are too many hyperparameters to be tuned, including but not limited to total number of layers, number of layers for each stack, etc. Earlier works mostly followed a paradigm established by AlexNet and VGG<sup>17</sup>, where the construction of the network (along with design of its building blocks) is coupled with the scaling of the baseline architecture. Both process were done mannually and heuristically. As the block-level innovations slowly stagnitized, a more effective and hollistic approach to design and scale a network is needed to push performance further. NAS<sup>13</sup> provided a solution. Instead of constructing a network mannually from scratch, a neural architecture search is performed over a pre-defined search space comprising of unit-level and/or block-level cells as the building bricks to automatically construct a network layer-by-layer or block-by-block. It is natural to combine a well-trained NAS controller with a well-defined search space to produce a highly competitive baseline architecture. Followed by either automatic or mannual scaling schemes, such a baseline could then be scaled up/down to suit various resource budgets while maintaining its competitiveness. Adopting this design paradigm, works such as MnasNet<sup>25</sup> and its successor the EfficientNets<sup>26</sup> have produced state-of-the-art convolutional networks across a wide of range of complexities.

Although the network designs have progressed significantly, the most basic convolutional operation remains the same. In modern frameworks such as [PyTorch](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html), aside from the kernel sizes and channels, a convolutonal module could be customized by specifying additional parameters, including:

* stride: controls the striding of the convolutional operation over a feature map
* padding: specifies the additional pixels padded on the edges of the feature map before convolving
* dilation: dilated convolution increases the receptive field of the filter without increasing the kernel size
* groups: grouped convolution could reduce computational complexity; an extreme situation is groups=channels, in which case it became depthwise

In modern network design, to alleviate the burden of dimension mismatch between skip connections, it is desirable to customize the convolutional modules such that they preserve feature map dimensions exactly when stride is 1. For an input feature map of size $$[H_{in}, W_{in}]$$, a conv-2d filter would produce an output feature map of the size:


$$
\begin{split}
H_{out} &= \frac{H_{in}+2*padding[0]-dilation[0]*(kernel\_size[0]-1)-1}{stride[0]}+1\\
W_{out} &= \frac{W_{in}+2*padding[1]-dilation[1]*(kernel\_size[1]-1)-1}{stride[1]}+1
\end{split}
$$


For instance, if the kernel size is 5, then setting the padding=dilation=2 would ensure the preservation of dimensions. Such a configuration would also exactly halve the feature maps if stride=2, another desirable property, as setting stride=2 is the most common downsampling technique employed in modern networks.

Another interesting anecdote about convolutional filters is that they are actually not convolutional at all. The proper name for the actual operation that they do perform should in fact be [auto-correlation](https://en.wikipedia.org/wiki/Autocorrelation#:~:text=Autocorrelation%2C%20also%20known%20as%20serial,the%20time%20lag%20between%20them.), borrowed from the field of statistics. The difference is very subtle: auto-correlation became the true convolution if the input feature maps are transposed first, which of course is not necessary. Nonetheless, the name "convolution", just as its status as the corner-stone of modern computer vision networks, have stood the test of time.

The sections below would now introduce the basic operations of a 2D convolutional filter, followed by 1D, 3D and separable convolutional filters.

* __forward-pass__

$$
\begin{split}y_{s,i,j}&=x_{p,i,j}*w_{s,p,i',j'}=\sum\limits_{p}^C\sum\limits_{i'}^K\sum\limits_{j'}^Kx_{p,i-i',j-j'}\cdot{w_{s,p,i',j'}}\\s,p&\text{: output/input channel index}\\i,j,k&\text{: activation index}\\K&\text{: kernel size}\\H&\text{: fmap size}\\C&\text{: input channels}\\M&\text{: output channels}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i,j}}}&=\sum\limits_{s}^Mw_{s,p,i,j}^T*\frac{\partial{L}}{\partial{y_{s,i',j'}}}=\sum\limits_{s}^M\sum\limits_{i'}^H\sum\limits_{j'}^Hw_{s,p,i-i',j-j'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\end{split}
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{s,p,i,j}}}&=x_{p,i,j}^T*\frac{\partial{L}}{\partial{y_{s,i',j'}}}=\sum\limits_{i'}^H\sum\limits_{j'}^Hx_{p,i-i',j-j'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i',j'}}}}\end{split}
$$

##### Conv-1D

* __forward-pass__

$$
\begin{split}y_{s,i}&=x_{p,i}*w_{s,p,i'}=\sum\limits_{p}^C\sum\limits_{i'}^Kx_{p,i-i'}\cdot{w_{s,p,i'}}\end{split}
$$

* __backward-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{x_{p,i}}}&=\sum\limits_{s}^Mw_{s,p,i}^T*\frac{\partial{L}}{\partial{y_{s,i'}}}=\sum\limits_{s}^M\sum\limits_{i'}^Hw_{s,p,i-i'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i'}}}}\end{split}\\w^T\text{: rotate }180^o
$$

* __parameter-pass__

$$
\begin{split}\frac{\partial{L}}{\partial{w_{s,p,i}}}&=x_{p,i}^T*\frac{\partial{L}}{\partial{y_{s,i'}}}=\sum\limits_{i'}^Hx_{p,i-i'}^T\cdot{\frac{\partial{L}}{\partial{y_{s,i'}}}}\end{split}\\x^T\text{: rotate }180^o
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

##### SeparableConv-2D

Separable convolutions break a standard convolution into two steps: a depthwise convolution focusing on each channel individually, followed by a point-wise operation across the channels. Note that while the following mathematical expressions treated the separable convolution as a single module, in practice it is common to implement this operation as a stack of two seperate layers, for instance in PyTorch:

```python
def forward(self, x):
    """ forward method without dropout """

    x = self.relu(self.bn1(self.convdw1(x)))	# convdw1: a depthwise layer by setting groups = inplanes = outplanes
    x = self.relu(self.bn2(self.conv2(x)))		# conv2: a 1x1 pointwise layer
    return x
```

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

Batch Normalization (BN)<sup>15</sup> is the standard practice for convolutional networks, employed by most if not all of the major models. It works by approximately normalizing the activations of a layer into a normal distribution with zero mean and unit variance for each of its channels. The original concept of reducing covariate shift had not yet been fully understood or accepted as the reason for why BN works so well. Batch Normalization technique is tightly related to the practice of mini-batching, a standard method that feeds the network with batches of data samples and trains with stochastic gradient descent on the entirety of the batch. It uses the mini-batch statistics as estimates to the true means and variances of the distributions of activations. Two learnable scaling and shifting factors control the module's final behavior and are trained along with the network parameters.

The coupling with mini-batch setting provided BN with computationally efficient estimation for the distributions, but it also limited its application to other settings, for instance when the batch size is too small or when the model is trained fully stochastically. Particularly for sequence models, including RNN's and Transformers<sup>cite</sup>, Layer Normalization (LN)<sup>27</sup> is a more popular choice. LN works in the same way as BN, except that it now estimates the true mean and variance of distributions through per-layer statistics (the activations) instead of per-minibatch. The most significant application for LN is perhaps the multi-head self-attention modules<sup>cite</sup>, the counterpart of convolution filters in modern Transformer networks.

The vanilla Layer Normalization combines the activations from all channels in that layer. This makes it an undesirable choice for convolutional networks, as  it essentially altered the network's expressiveness by providing additional cross-channel information to its layers. Furthermore, generally the feature maps are gradually downsampled through the network; the shallow layers have large activation maps and thus expensive Layer Norm computations, while the deeper layers have very small activation maps to produce any meaningful estimation statistically. To bridge the gap between LN and BN, several subsequent works have been proposed. Instance Normalization (IN)<sup>28</sup> and Group Normalization (GN)<sup>29</sup> are also based on the same idea of normalizing a layer's activations by estimating the true mean and variance of the distribution, but they differ in how the estimations are produced: IN uses per-channel statistics, thus removing the cross-channel information in LN; GN seeks the middle ground between IN and LN by using statistics from a group of multiple channels. Another notable but less commonly used method is Weight Normalization (WN)<sup>30</sup>, which works by normalizing the layer weights instead of the layer activations (computationally cheaper, as there are usually much more activations than weights for a given layer).

Since most of the commonly used normalization methods are variants of Batch Normalization, here its modular formulation is included, following the definitions used in the original paper by Sergey Ioffe and Christian Szegedy.

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





#### Bibliography

1. Ian Goodfellow, Yoshua Bengio, Aaron Courvill, "Deep Learning"

#### Online Materials

1. Simon Osindero, ["Lecture-03: Neural Network Foundations"](https://www.youtube.com/watch?v=5eAXoPSBgnE&list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs&index=3), the DeepMindxUCL lectures 2018
2. Jefkine, ["Backpropagation In Convolutional Neural Networks"](https://canvas.stanford.edu/files/1041875/download?download_frd=1&verifier=tFv4Jc7bCezxJg9rG2yhEKEERi70zJ3ScmFbNlbN) 
3. Christopher Olah, ["Understanding LSTM networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
4. Van Huyen DO, ["Back propagation in Long Short Term Memory (LSTM)"](https://medium.com/@dovanhuyen2018/back-propagation-in-long-short-term-memory-lstm-a13ad8ae7a57)

#### References

5. Xavier Glorot et al, "[Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)", AISTATS 2011
2. Andrew L. Maas et al, "[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)", ICML 2013
3. Djork-Arn´e Clevert et al, "[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)", ICLR 2016
4. Günter Klambauer et al, "[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)", NeurIPS 2017
5. Dan Hendrycks et al, "[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)", Arxiv 2018
6. Kaiming He et al, "[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)", ICCV 2015
7. Nitish Srivastava et al, "[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)", JMLR 2014
8. Frank Rosenblatt, "[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf)", 1958
9. Yan LeCun et al, ["Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)", Neural Computation 1989
10. Dan C.Ciresan et al, "[Flexible, High Performance Convolutional Neural Networks for Image Classification](http://people.idsia.ch/~juergen/ijcai2011.pdf)", IJCAI 2011
11. Alex Krizhevsky et al, "[ImageNet Classiﬁcation with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)", NeurIPS 2012
12. Kaiming He et al, "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)", CVPR 2016
13. Barret Zoph & Quoc V.Le, "[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)", ICLR 2017
14. Christian Szegedy et al, "[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)", CVPR 2016
15. Sergey Ioffe & Christian Szegedy, "[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)", ICML 2015
16. Francois Chollet, "[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)", CVPR 2017
17. Karen Simonyan & Andrew Zisserman, "[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)", ICLR 2015
18. Saining Xie et al, "[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)", CVPR 2017
19. Sergey Zagoruyko & Nikos Komodakis, "[Wide Residual Networks](https://arxiv.org/abs/1605.07146)", Arxiv 2016
20. Andrew G.Howard et al, "[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)", CoRR 2017
21. Mark Sandler et al, "[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)", CVPR 2018
22. Xiangyu Zhang et al, "[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)", CVPR 2018
23. Ningning Ma et al, "[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)", EECV 2018
24. Gao Huang et al, "[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)", CVPR 2017
25. Mingxing Tan et al, "[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)", CVPR 2019
26. Mingxing Tan & Quoc V.Le, "[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)", ICML 2019
27. Jimmy Lei Ba et al, "[Layer Normalization](https://arxiv.org/abs/1607.06450)", Arxiv 2016
28. Dmitry Ulyanov et al, "[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)", Arxiv 2017
29. Yuxin Wu & Kaiming He, "[Group Normalization](https://arxiv.org/abs/1803.08494)", EECV 2018
30. Tim Salimans et al, "[Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://papers.nips.cc/paper/6114-weight-normalization-a-simple-reparameterization-to-accelerate-training-of-deep-neural-networks.pdf)", NeurIPS 2016