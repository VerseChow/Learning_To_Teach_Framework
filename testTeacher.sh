#!/bin/bash
set -x
set -e

MODEL=$1

case $MODEL in
  CIFAR)
    cd ./CIFAR_Agent
    sudo python3 cifar_train.py --teacher_flg False
  ;;
  MNIST)
		cd ./MNIST_Agent
		sudo python3 mnist_train.py --teacher_flg False
  ;;
  *)
    echo "Please specify CIFAR or MNIST agent"
    exit
    ;;
esac
cd ..

set +x
set -x