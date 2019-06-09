#!/bin/sh

ln -s ../test_images/ ./test_images
cp ./src/fps_main.cc ./src/main.cc
make clean
make
mv LeNet fps_LeNet

echo "./LeNet 1"
./fps_LeNet 1
echo "./LeNet 2"
./fps_LeNet 2
echo "./LeNet 3"
./fps_LeNet 3
echo "./LeNet 4"
./fps_LeNet 4
echo "./LeNet 5"
./fps_LeNet 5
echo "./LeNet 6"
./fps_LeNet 6
