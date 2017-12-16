# Learning To Teach Framework
This is the repo for reproducing the paper called [Leaning to Teach](https://openreview.net/forum?id=HJewuJWCZ&noteId=HJewuJWCZ) for EECS498 Reinforcement Learning final project.

## Computer Environment Requirement
### Python Version
1. We use python3.5 on ubuntu16.04

### Dependency
1. Our framework is based on tensorflow r1.4. Please follow [Installation Instruction](https://www.tensorflow.org/install/) to install tensorflow

## Basic Usage
Our pipleline is for training a teacher agent to guide a student agent learn more quickly. In this case, you have to train a teacher agent then test the teacher agent to train a student model. We have two options to train teacher agent. You can use either MNIST or CIFAR dataset to train teacher agent as following guide

### Train Teacher Agent
1. run `./trainTeacher.sh <dataset>`, the option of dataset is `CIFAR` or `MNIST`
2. trained weight is saved in the folder `<dataset>_Agent/pretrained_weight_for_teacher`

### Test Teacher Agent
1. run `./testTeacher.sh <dataset>`, the option of dataset is `CIFAR` or `MNIST`
2. make sure in the folder `<dataset>_Agent/pretrained_weight_for_teacher` contains pretrained weight
3. training will stop as long as student gets to 90% accuracy

## Advanced Usage
For advanced usage, you can go to `CIFAR_Agent` or `MNIST_Agent` folder to check the detailed code