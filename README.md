# Learning To Teach Framework
This is the repo for reproducing the paper called [Leaning to Teach](https://openreview.net/forum?id=HJewuJWCZ&noteId=HJewuJWCZ) for EECS498 Reinforcement Learning final project.

# Usage
## Train Teacher Agent
1. run `./trainTeacher.sh <dataset>`, the option of dataset is `CIFAR` or `MNIST`
2. trained weight is saved in the folder `<dataset>_Agent/pretrained_weight_for_teacher`

## Test Teacher Agent
1. run `./testTeacher.sh <dataset>`, the option of dataset is `CIFAR` or `MNIST`
2. make sure in the folder `<dataset>_Agent/pretrained_weight_for_teacher` contains pretrained weight