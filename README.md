# Deep Companion Learning: Enhancing Generalization Through Historical Consistency  (ECCV 2024)
[Ruizhao Zhu](https://ruizhaoz.github.io/) and [Venkatesh Saligrama](https://sites.bu.edu/data/). Boston University.

[//]: # (![]&#40;globaldrive.png&#41;)
<img src="globaldrive.png" alt="drawing" width="500"/>

This is official PyTorch/GPU implementation of the paper [Deep Companion Learning: Enhancing Generalization Through Historical Consistency](https://arxiv.org/abs/2407.18821):

```
@inproceedings{zhu2024deep,
  title={Deep Companion Learning: Enhancing Generalization Through Historical Consistency},
  author={Zhu, Ruizhao and Huang, Peng and Ohn-Bar, Eshed and Saligrama, Venkatesh},
  booktitle={ECCV},
  year={2024}
}
```

## Updates
[09/11/2024] Initial Update!


## Getting Started
* To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.
* We run our model on CARLA 0.9.13, install and environment needed [here](https://github.com/carla-simulator/carla/releases).
* Please follow requirement.txt to setup the environment.

## Dataset Preparation
* ### CIFAR100 dataset
* ### Tiny-imageNet
You can and download the Tiny-Imagenet Dataset from the [here]([https://www.argoverse.org/av2.html#download-link](https://github.com/rmccorm4/Tiny-Imagenet-200)). official website. 
* ### ImageNet-1k



## Training and testing
```bash
./runner.sh
```
