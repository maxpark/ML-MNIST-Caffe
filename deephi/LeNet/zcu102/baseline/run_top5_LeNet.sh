#!/bin/sh

ln -s ../test_images/ ./test_images
cp ./src/top5_main.cc ./src/main.cc
make clean
make
mv LeNet top5_LeNet
./top5_LeNet 1
