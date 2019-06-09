import os


MNIST_DIR     = os.environ["ML_DIR"]             # MNIST working dir
WORK_DIR       = MNIST_DIR + "/caffe"              # MNIST caffe dir

# environmental variables: $CAFFE_ROOT and $CAFFE_TOOLS_DIR
CAFFE_ROOT= os.environ["CAFFE_ROOT"]             # MNIST working dir

CAFFE_TOOLS_DIR= CAFFE_ROOT  + "/distribute"             # the effective Caffe root

# project folders
MNIST_JPG_DIR      = MNIST_DIR + "/input/mnist_jpg"   # where plain MNIST JPEG images are placed
INPUT_DIR          = MNIST_DIR + "/input"             # input image and databases main directory
LMDB_DIR           = INPUT_DIR + "/lmdb"              # where validation and training LMDB databases are placed
VALID_DIR          = LMDB_DIR + "/valid_lmdb"  # i.e. "/home/danieleb/ML/mnist/input/lmdb/valid_lmdb"
TRAIN_DIR          = LMDB_DIR + "/train_lmdb"  # i.e. "/home/danieleb/ML/mnist/input/lmdb/train_lmdb"

#project file for mean values
MEAN_FILE = INPUT_DIR + "/mean.binaryproto" # i.e. "/home/danieleb/ML/mnist/input/mnist_mean.binaryproto"

