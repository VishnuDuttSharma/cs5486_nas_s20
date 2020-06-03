__author__ = "Vishnu Dutt Sharma"
__description__ = "Template for running regression experiments"

# Import reuired libraries
import sys
from regression_utils import *

#######################################################################
######################## User Options #################################
#######################################################################

# Dataset name. Options are 'boston_housing', 'breast_cancer'
DATASET_NAME = 'breast_cancer'
# Batch Size. Set to None if using LBFGS
batch_size=8
# Model type. Options are DNN3, DNN4, DNN5
model_type = 'DNN3'
# Whether to use dropout or not (set to False is not using dropout)
dropstate=True
# model name is same as the file name by default. Change the string if other name is required 
SAVE_PATH = f'models/{sys.argv[0]}'
# Optimizer to be used. Options are 'sgd', 'lbfgs', 'kfac'
optim_name = 'sgd'
# Learning rate
lr = 0.01

#######################################################################
#######################################################################
#######################################################################



# Choose device (CPU/GPU) as per availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get data utilities. 
dataset = Dataset(name=DATASET_NAME, batch_size=batch_size)
# Creating model
net=None
if model_type == 'DNN3':
    net = DNN3(len(dataset.mean_X));
elif model_type == 'DNN4':
    net = DNN4(len(dataset.mean_X));
else:
    net = DNN5(len(dataset.mean_X));
# Print the model summary
net.to(device);
# Send model to device (GPU/CPU)
if DATASET_NAME == 'boston_housing':
    print(summary(net, (13,)))
elif DATASET_NAME == 'breast_cancer':
    print(summary(net, (30,)))

# Use Mean-squared error loss for classification
criterion = nn.MSELoss()

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
net = train_model(net, optimizer, criterion, dataset, SAVE_PATH, device)
# Get training accuracy
training_acc(net, dataset, SAVE_PATH, device)
# Get test accuracy
get_accuracy(net, dataset, SAVE_PATH, device)


