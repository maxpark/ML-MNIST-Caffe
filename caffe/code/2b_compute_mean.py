# ##################################################################################################
# USAGE
# python 2b_compute_mean.py

# It reads the images from LMDB training database and create the mean file

# by daniele.bagni@xilinx.com

# ##################################################################################################

import os
import glob
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from config import mnist_config as config

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import caffe
from caffe.proto import caffe_pb2
import lmdb

import argparse

# ##################################################################################################
# working directories

INP_DIR  = config.INPUT_DIR                              # "/home/ML/mnist/input"
INP_LMDB = config.LMDB_DIR + "/train_lmdb"               # "/home/ML/mnist/input/lmdb/train_lmdb"


# ##################################################################################################
# MEAN of all training dataset images

print ('\nGenerating mean image of all training data')
mean_command =  config.CAFFE_TOOLS_DIR + "/bin/compute_image_mean.bin -backend=lmdb "

os.system(mean_command + INP_LMDB + '  ' + config.MEAN_FILE)


# ##################################################################################################
# show the mean image

blob = caffe.proto.caffe_pb2.BlobProto()
data  = open(config.MEAN_FILE).read()
blob.ParseFromString(data)

mean_array = np.asarray(blob.data, dtype=np.float32).reshape((blob.channels, blob.height, blob.width))
print " mean value channel 0: ", np.mean(mean_array[0,:,:])
print " mean value channel 1: ", np.mean(mean_array[1,:,:])
print " mean value channel 2: ", np.mean(mean_array[2,:,:])

'''
# display image of mean values
arr = np.array(caffe.io.blobproto_to_array(blob))[0, :, :, :].mean(0)
plt.imshow(arr, cmap=cm.Greys_r)
#plt.imshow(arr, cmap=cm.brg)
plt.show()
'''

'''
I0424 11:36:44.385044 19738 db_lmdb.cpp:35] Opened lmdb /home/danieleb/ML/mnist/input/lmdb/train_lmdb
I0424 11:36:44.385272 19738 compute_image_mean.cpp:70] Starting iteration
I0424 11:36:44.409319 19738 compute_image_mean.cpp:95] Processed 10000 files.
I0424 11:36:44.433202 19738 compute_image_mean.cpp:95] Processed 20000 files.
I0424 11:36:44.457010 19738 compute_image_mean.cpp:95] Processed 30000 files.
I0424 11:36:44.480773 19738 compute_image_mean.cpp:95] Processed 40000 files.
I0424 11:36:44.503792 19738 compute_image_mean.cpp:95] Processed 50000 files.
I0424 11:36:44.503803 19738 compute_image_mean.cpp:108] Write to /home/danieleb/ML/mnist/input/mean.binaryproto
I0424 11:36:44.503932 19738 compute_image_mean.cpp:114] Number of channels: 3
I0424 11:36:44.503939 19738 compute_image_mean.cpp:119] mean_value channel [0]: 33.6828
I0424 11:36:44.503960 19738 compute_image_mean.cpp:119] mean_value channel [1]: 33.6828
I0424 11:36:44.503968 19738 compute_image_mean.cpp:119] mean_value channel [2]: 33.6828
 mean value channel 0:  33.6828
 mean value channel 1:  33.6828
 mean value channel 2:  33.6828
'''
