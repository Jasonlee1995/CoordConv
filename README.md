# CoordConv Implementation with Pytorch
- Unofficial Pytorch implementation of the paper *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```
- Using Single GPU


## 1. Implementation Details
- data : generate regression dataset
- dataset.py : regression dataset
- main.py : run train with argparse
- model.py : regression models used in paper
- train.py : train, test, inference model
- Check AddCoord.ipynb : check CoordConv
- Visualize - Figures (CoordConv).ipynb : visualize 100 figures from CoordConv models per hyperparameter settings
- Details
  * Official codes are weird cause there are no activation functions on CoordConv model
  * No learning rate scheduler for convenience


## 2. Reference
- An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution [[paper](https://arxiv.org/pdf/1807.03247.pdf)] [[official code](https://github.com/uber-research/CoordConv)]
