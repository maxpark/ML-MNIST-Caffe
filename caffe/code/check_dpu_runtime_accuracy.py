# USAGE
# python /home/ML/mnist/caffe/code/check_dpu_runtime_accuracy.py -i /home/ML/mnist/deephi/LeNet/quantiz/zcu102/rpt/logfile_top5_LeNet.txt 

# It checks the top-1 and top-5 accuracy obtained at runtime by DeePhi DPU, by analysis of the related logfile 


# by daniele.bagni@xilinx.com

# ##################################################################################################

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from datetime import datetime
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--file",  required=True, help="input logfile")
ap.add_argument("-n", "--numel", default="1000", help="number of test images")
args = vars(ap.parse_args())

logfile = args["file"] # root path name of dataset

try:
    f = open(logfile, "r")
except IOError:
    print 'cannot open ', logfile
else:
    lines = f.readlines()
    tot_lines = len(lines)
    print logfile, ' has ', tot_lines, ' lines'

#f.seek(0)
f.close()
    
# ##################################################################################################

NUMEL = int(args["numel"]) #1000

# initialize the label names for the CIFAR-10 dataset
labelNames = {"airplane" :0 , "automobile" :1, "bird" :2, "cat" :3, "deer" :4, "dog" :5,
              "frog" :6, "horse" :7, "ship" :8 , "truck" :9 }
       
# ##################################################################################################

top1_true  = 0
top1_false = 0
top5_true  = 0
top5_false = 0
img_count  = 0
false_pred = 0

test_ids = np.zeros(([NUMEL,1]))
preds    = np.zeros(([NUMEL, 1]))
idx = 0

for ln in range(0, tot_lines):

    if "DBG" in lines[ln]:
        
        top5_lines = lines[ln:ln+6]

	filename= top5_lines[0].split("test_images/")[1]
        s2 = filename.index("_")
        class_name = filename[: s2].strip()
        #print 'DBG: found class ', class_name, ' in line ', ln, ': ', lines[ln]

	predicted = top5_lines[1].split("name = ")[1].strip()

        if class_name in top5_lines[1]:
            top1_true += 1
            top5_true += 1
        elif class_name in top5_lines[2]:
            top5_true += 1
            top1_false +=1
        elif class_name in top5_lines[3]:
            top5_true += 1
            top1_false +=1            
        elif class_name in top5_lines[4]:
            top5_true += 1
            top1_false +=1            
        elif class_name in top5_lines[5]:
            top5_true += 1
            top1_false +=1            
        else:
            top5_false += 1            
            top1_false +=1
            
        test_ids[idx] = labelNames[class_name] # ground truth 
        preds[idx]    = labelNames[predicted ] # actual prediction
        
        if (predicted != class_name) :
            print "LINE: ", filename #top5_lines[0].split("./")[1].strip()
            print "PREDICTED: ", preds[idx], predicted
            print "EXPECTED : ", test_ids[idx], class_name
            for k in range(1, 6):
                print top5_lines[k].strip()            
            print "\n"


        img_count +=1
        idx += 1

        if ( idx == (NUMEL-1) ):
            break
            
            
    else:
        continue
            


assert (top1_true+top1_false)  == img_count, "ERROR: top1 true+false not equal to the number of images"
assert (top5_true+top5_false)  == img_count, "ERROR: top5 true+false not equal to the number of images" 

print 'number of total images predicted ', img_count
print 'number of top1 false predictions ', top1_false
print 'number of top1 right predictions ', top1_true
print 'number of top5 false predictions ', top5_false
print 'number of top5 right predictions ', top5_true

top1_accuracy = float(top1_true)/(top1_true+top1_false) 
top5_accuracy = float(top5_true)/(top5_true+top5_false) 

print('top1 accuracy = %.2f' % top1_accuracy)
print('top5 accuracy = %.2f' % top5_accuracy)


