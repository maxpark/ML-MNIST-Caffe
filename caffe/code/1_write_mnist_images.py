# USAGE
# python 1_write_mnist_images.py
# (optional) --pathname $HOME/ML/mnist/input/mnist_jpg

# It downloads the MNIST dataset from KERAS library and put it into JPG images organized in 3 folders:
# train (50000 images) validation (9000 images) and test (1000 images) with their proper labels txt files.
#
# It also builds a 4th directory for Calibration during the Qunatization process with DeePhi DECENT tool.

# by daniele.bagni@xilinx.com

# ##################################################################################################

# set the matplotlib backend before any other backend, so that figures can be saved in the background
import matplotlib     
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

# import the necessary packages
from config import mnist_config as config
from keras.datasets import mnist 
from datetime import datetime
import matplotlib.pyplot as plt 
import cv2
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pathname", default=config.MNIST_JPG_DIR, help="path to the dataset")
args = vars(ap.parse_args())


path_root = args["pathname"] # root path name of dataset

if (not os.path.exists(path_root)): # create "path_root" directory if it does not exist
    os.mkdir(path_root)


# ##################################################################################################

# load the training and validation data from MNIST dataset, then scale it into the range [0, 1]
print("[INFO] loading MNIST data...")
((trainX, trainY ), (validX, validY )) = mnist.load_data()
trainX = trainX.astype("float") / 255.0
validX = validX.astype("float") / 255.0


# initialize the label names for the CIFAR-10 dataset
labelNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# ##################################################################################################
# BUILD THE VALIDATION SET with last 5000 images of validX

wrk_dir = path_root + "/val"

if (not os.path.exists(wrk_dir)): # create "val" directory if it does not exist
    os.mkdir(wrk_dir)

f_test = open(wrk_dir+"/test.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir+"/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()

val_count = 0

vcounter = [0,0,0,0,0,0,0,0,0,0]

testX=validX[5000:] #last 5000 images
testY=validY[5000:] #last 5000 labels
a = np.arange(0, len(testX))

for i in a :
    image = testX[int(i)]
    image2 = image *  255.0
    image2 = image2.astype("uint8")

    val_count = val_count+1

    vcounter[ int(testY[int(i)]) ] =     vcounter[ int(testY[int(i)]) ] +1;

    string = "%05d" % vcounter[ int(testY[int(i)]) ]

    class_name = labelNames[int(testY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name) #https://github.com/BVLC/caffe/issues/3698

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string = " %1d" % int(testY[int(i)]) 
    f_test.write(path_name + string + "\n")

    #cv2.imwrite(path_name, image2)
    rgb_image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path_name, rgb_image2)
    
    print(path_name)

f_test.close()



# ##################################################################################################
# BUILD THE TEST SET with first 5000 images of validX


wrk_dir = path_root + "/test"

if (not os.path.exists(wrk_dir)): # create "test" directory if it does not exist
    os.mkdir(wrk_dir)

f_test  = open(wrk_dir+"/test.txt", "w")   #open file test.txt"
f_test2 = open(wrk_dir+"/test2.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir+"/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
tcounter = [0,0,0,0,0,0,0,0,0,0]

testX=validX[0:5000] #first 5000 images
testY=validY[0:5000] #first 5000 labels
a = np.arange(0, len(testX))

test_count = 0
test2_count = -1

for i in a : 
    image = testX[int(i)]
    #image2= cv2.resize(image, (32,28), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("uint8")

    tcounter[ int(testY[int(i)]) ] =     tcounter[ int(testY[int(i)]) ] +1;

    test_count = test_count +1
    test2_count = test2_count +1
    string = "%05d" % tcounter[ int(testY[int(i)]) ]

    class_name = labelNames[int(testY[int(i)])]
  
    path_name = wrk_dir + "/" + class_name + "_" + string + ".jpg"

    string2 = " %1d" % test2_count
    f_test.write(path_name + string2 + "\n")
    f_test2.write(class_name + "_" + string + ".jpg" + string2 + "\n")
    
    #cv2.imwrite(path_name, image2)
    rgb_image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path_name, rgb_image2)
    
    #cv2.imshow(labelNames[int(testY[int(i)])], rgb_image2)
    #cv2.waitKey(0)
    
    print(path_name)

f_test.close()
f_test2.close()


# ##################################################################################################
# BUILD THE TRAIN SET of 60000 IMAGES

wrk_dir = path_root + "/train"

if (not os.path.exists(wrk_dir)): # create "train" directory if it does not exist
    os.mkdir(wrk_dir)

f_test = open(wrk_dir + "/train.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir + "/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
counter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, len(trainX))

for i in a : 
    image = trainX[int(i)]
    #image2= cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("uint8")
   
    counter[ int(trainY[int(i)]) ] =     counter[ int(trainY[int(i)]) ] +1;
    string = "%05d" % counter[ int(trainY[int(i)]) ]

    class_name = labelNames[int(trainY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name)

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string = " %1d" % int(trainY[int(i)]) 
    f_test.write(path_name + string + "\n")

    #cv2.imwrite(path_name, image2)
    rgb_image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path_name, rgb_image2)    

    #cv2.imshow(labelNames[int(testY[int(i)])], rgb_image2)
    #cv2.waitKey(0)
    
    #print(path_name)

f_test.close()

# ##################################################################################################
# BUILD THE CALIBRATION IMAGES SET from first 2000 images of training dataset

wrk_dir = path_root + "/calib"

if (not os.path.exists(wrk_dir)): # create "calibration" directory if it does not exist
    os.mkdir(wrk_dir)

f_calib = open(wrk_dir + "/calibration.txt", "w")   #open file calibration.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s]
    
ccounter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, len(trainX[0:2000]))

calib_count = -1
for i in a : 
    image = trainX[int(i)]
    #image2= cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("uint8")
   
    ccounter[ int(trainY[int(i)]) ] =     ccounter[ int(trainY[int(i)]) ] +1;

    calib_count = calib_count + 1
    string = "%05d" % ccounter[ int(trainY[int(i)]) ]

    class_name = labelNames[int(trainY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name)

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string2 = " %1d" % int(calib_count) 
    f_calib.write(class_name + "/" + class_name + "_" + string + ".jpg" + string2 + "\n")

    #cv2.imwrite(path_name, image2)
    rgb_image2 = cv2.cvtColor(image2,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(path_name, rgb_image2)

    #cv2.imshow(labelNames[int(testY[int(i)])], rgb_image2)
    #cv2.waitKey(0)
    
    print(path_name)

f_calib.close()

################################################################################

print "   histogram of validation dataset classes", vcounter
print "np.histogram of validation dataset classes", np.histogram(testY)[0]
print "   histogram of test dataset classes", tcounter
print "np.histogram of test dataset classes", np.histogram(testY)[0]
print "   histogram of train dataset classes", counter
print "np.histogram of train dataset classes", np.histogram(trainY)[0]
print "   histogram of calib dataset classes", ccounter
print "np.histogram of calib dataset classes", np.histogram(trainY[0:2000])[0]

print "###################################################################"
print "SUMMARY"
print "###################################################################"
              
print "number of images in the original Training   dataset: ", len(trainX)
print "number of images in the original Validation dataset: ", len(validX)      

print "adopted Test        set contains ", test_count,    " images"
print "adopted Validation  set contains ", val_count,     " images"
print "adopted Calibration set contains ", calib_count+1, " images"
print("END\n")

