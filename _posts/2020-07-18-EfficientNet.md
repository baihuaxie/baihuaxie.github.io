---
layout: post
comment: true
title: EfficientNet: One Net To Rule Them All?
date: 2020-07-18
author: Baihua Xie
tags: cnn cv nas
mins: 10
---

> This blog post briefly introduces the paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Mingxing Tan & Quoc V.Le, ICML 2019. In the paper, the authors detailed a paradigm to design convolutional neural networks based on an augmented form of neural architecture search, where the search space is reduced considerably by fixing the network as a sequence of blocks each with predefined search space to search layers at a block rather than network level. A baseline architecture is produced, which is subsequently modified by a compound scaling scheme proposed in the paper to simultaneously scale the model in depths, widths and resolution dimensions in a unified manner. The result is a set of networks termed EfficientNets, widely ranging in their complexity, which achieve state-of-the-art or highly competitive results against mainstream benchmark models under comparable resource constraints in classificiation and transfer tasks.

Modern convolutional neural networks typically are developed and benchmarked under a fixed resource budget, commonly in the form of limiting the number of trainable parameters in the network, the number of FLOPs required during inference of a sample, and in scenarios where the responsiveness of the model is critical such as models deployed on mobile platforms,  the latency or throughput of inference. It is therefore of considerable practical significance that a model can be designed to accomodate a resource constraint, and preferrably, to be adapted to a different set of constraints by simple scaling schemes without the need of re-design. 

It has long been established in the deep learning community that scaling up a model could lead to better performance on a given dataset. While earlier ground-breaking works such as VGG^2^ and ResNet^3^ mostly validated the benefits of increasing network depths, recent works have also begun to explore other dimensions for scaling, including model width^4^ and input resolution. As the authors noted in the paper, most of the previous works on network scaling has focused on scaling exactly one of the possible dimensions at a time. The authors argue that this could potentially limit the effect of model scaling; for instance, if model A takes a higher resolution of inputs while being as narrow in width and as shallow in depth as model B, intuitively model A could not have utilized the extra input resolution to benefit its performance. 

To scale a model to fit a given resource budget, a baseline architecture needs to be designed first. The paper opted for a NAS-based approach that is built upon a previous work by the same authors on neural architecture search for mobile platforms, termed MnasNet^5^, where a multi-objective reward that incorporates both accuracy and inference latency on target platform is calculated to train the controller network. 

As with any NAS method, composition of the search space is critical to the performance of algorithm. In the MnasNet paper, a search space primarily comprised of mobile-inverted-bottleneck layers^6^ is used for their superior performance of in building models for mobile platforms. Another factor is the search space size, as NAS methods are notoriously costly in training^7^. Instead of constructing a child network layer by layer, MnasNet first defines a skeleton network architecture comprising of a sequence of blocks. With this skeleton network, each block is associated with its own block-level search space, and only one search is required to produce a block of several layers (number of layers is also a searchable parameter in the search space, along with the type of the layer). By breaking the search procedure from searching on per-layer level to searching on per-block level, this alteration effectively reduces the overall search space size to produce a final baseline network of same depths by a factor of $$e^\frac{1}{N}$$, where N is the average number of layers per block in the network. 

#### Reference

1. Mingxing Tan & Quoc V.Le, ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946), ICML 2019
2. Karen Simonyan & Andrew Zisserman, ["Very Deep Convolutional Networks for Large Scale Image Recognition"](https://arxiv.org/abs/1409.1556), ICLR 2015
3. Kaiming He et al, ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385), CVPR 2016
4. Andrew G. Howard et al, ["MobileNets: Efﬁcient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861), CoRR 2017
5. Mingxing Tan et al, ["MnasNet: Platform-Aware Neural Architecture Search for Mobile"](https://arxiv.org/abs/1807.11626), CVPR 2019
6. Mark Sandler et al, ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381), CVPR 2018
7. Barret Zoph & Quoc V.Le, ["Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)", ICLR 2017 