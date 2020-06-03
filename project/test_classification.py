__author__ = "Vishnu Dutt Sharma"
__description__ = "Template for running classification experiments"

# Import required libraries
import sys
from classification_utils import *

#######################################################################
######################## User Options #################################
#######################################################################

# Dataset name. Options for name are 'cifar10', 'mnist', 'fashion_mnist'
DATASET_NAME = 'cifar10'
# Batch Size. Set to None, if using LBFGS
batchsize=32
# Model Type. Options are CIFAR_CNN, MNIST_CNN, CIFAR_CNN_nopool, MNIST_CNN_nopool, CIFAR_DNN, MNIST_DNN (MNIST_CNN/DNN is also used for Fashion MNIST)
model_type = 'CIFAR_CNN'
# Dropout rate. Set to 'None', if not using dropout
droprate = 0.2
# model names is same as the file name by default. Change the string if other name is required 
SAVE_PATH = f'models/{sys.argv[0]}'
# Optimizer to be used. Options are 'sgd', 'lbfgs', 'kfac'
optim_name = 'sgd'
# Learning rate
lr = 0.001

#######################################################################
#######################################################################
#######################################################################

# Choose device (CPU/GPU) as per availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get data utilities. 
trainloader, valloader, testloader = datasets(name=DATASET_NAME, batchsize=batchsize)
# Get dictionary containing class names. Options are 'cifar_classes', 'mnist_classes', 'fashion_mnist_classes'
if DATASET_NAME == 'cifar10':
    classes =  cifar_classes
elif DATASET_NAME == 'mnist':
    classes = mnist_classes
elif DATASET_NAME == 'fashion_mnist':
    classes = fashion_mnist_classes

# Model options are CIFAR_CNN, MNIST_CNN, CIFAR_DNN, MNIST_DNN (MNIST_CNN/DNN is also used for Fashion MNIST)
# If not using dropout, set 'dropout=None' or leave blank
if model_type == 'CIFAR_CNN':
    net = CIFAR_CNN(droprate=droprate);
elif model_type == 'CIFAR_CNN_nopool':
    net = CIFAR_CNN_nopool(droprate=droprate);
elif model_type == 'CIFAR_DNN':
    net = CIFAR_DNN(droprate=droprate);
elif model_type == 'MNIST_CNN':
    net = MNIST_CNN(droprate=droprate);
elif model_type == 'MNIST_CNN_nopool':
    net = MNIST_CNN_nopool(droprate=droprate);
elif model_type == 'MNIST_DNN':
    net = MNIST_DNN(droprate=droprate);

# Send model to device (GPU/CPU)
net.to(device);
# Print network structure
print(net)

# Use cross-entropy loss for classification
criterion = nn.CrossEntropyLoss()

# Setting optimizer as mentioned above
optimizer = None
if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
elif optim_name == 'lbfgs':
    optimizer = optim.LBFGS(net.parameters(), lr=lr)
elif optim_name == 'kfac':
    optimizer = optimizers.kfac.KFACOptimizer(net, lr=lr)
    optimizer.acc_stats = True

# Train the model
net = train_model(net, optimizer, criterion, trainloader, valloader, SAVE_PATH, device)
# Get training accuracy
training_acc(net, trainloader, SAVE_PATH, device)    
# Get test accuracy
get_accuracy(net, testloader, classes, SAVE_PATH, device)


