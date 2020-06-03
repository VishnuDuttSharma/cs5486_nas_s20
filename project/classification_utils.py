__author__ = "Vishnu Dutt Sharma"
__description__ = "Utilities for experiments on classifier networks"

import time
import random
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Extra module added to print the summary of the model
from torchsummary import summary

### Configurations for deterministic results ######
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### Adding K-FAC library ot path
import sys
sys.path.append('KFAC_Pytorch/')
from KFAC_Pytorch import optimizers

####################################################################################
############################# Helper Variables #####################################
####################################################################################
### Names of the labels/classes for each dataset
# CIFAR-10
cifar_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# MNIST
mnist_classes = (0,1,2,3,4,5,6,7,8,9)

# Fashion MNIST
fashion_mnist_classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

###################################################################
######################## Data Utilities ###########################
###################################################################
def datasets(name, batchsize=None):
    """
    Function to create dataloaders for each dataset
    
    `name`: Name of the dataset. Allowed names: cifar10, mnist, fashion_mnist
    `batchsize`: Batch Size. If None is given as input, whole dataset is use (no mini-batch)
    """
    #### CIFAR-10 #####
    if name == 'cifar10':
        # Data tranformer: Tensor converter and Normalization
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        ## Downloading/loading the datasets
        # training and validation data
        trainvalset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        # Keep last 10000 as validation, rest as training data
        trainset, valset = torch.utils.data.random_split(trainvalset, [40000, 10000])
        
        # Test data
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
    #### MNIST ####
    elif name == 'mnist':
        # Data tranformer: Tensor converter and Normalization
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        
        ## Downloading/loading the datasets
        # training and validation data
        trainvalset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        # Keep last 10000 as validation, rest as training data
        trainset, valset = torch.utils.data.random_split(trainvalset, [50000, 10000])
        
        # Test data
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    #### Fashion MNIST ####
    elif name == "fashion_mnist":
        # Data tranformer: Tensor converter and Normalization
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

        ## Downloading/loading the datasets
        # training and validation data
        trainvalset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=transform)
        # Keep last 10000 as validation, rest as training data
        trainset, valset = torch.utils.data.random_split(trainvalset, [50000, 10000])
        
        # Test data
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                               download=True, transform=transform)
    else:
        print('Invalid dataset name. Allowed names: cifar10, mnist, fashion_mnist')
        return None
        
    print(f'Loading {name} dataset')
    
    ## Batch Size
    if batchsize == None:
        train_bs = len(trainset)
        val_bs = len(valset)
        test_bs = len(testset)
    else:
        train_bs = 32
        val_bs = 1024 # Setting higher values for faster inferece
        test_bs = 1024
        
    
    print(f'BatchSize: train:{train_bs}, val:{val_bs}, test:{test_bs}')
    
    
    # Create dataloaders/itratores
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_bs,
                                              shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
                                             shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

#########################################################################
######################### Network Architectures #########################
#########################################################################

### CIFAR Models ###
class CIFAR_CNN(nn.Module):
    """
    CNN based model for CIFAR
    
    Schema: CNN->ReLU->Pool --> CNN->ReLU->Pool --> Flatten -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(CIFAR_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
        if droprate is not None:
            self.apply_drop = True
            self.drop1 = nn.Dropout2d(droprate)
            self.drop2 = nn.Dropout2d(droprate)
            self.fc_drop1 = nn.Dropout(droprate)
            self.fc_drop2 = nn.Dropout(droprate)
        else:
            self.apply_drop = False

    def forward(self, x):
        """
        Forward Pass. CNN->ReLU->Pool --> CNN->ReLU->Pool --> Flatten -> FC->ReLU -> FC->ReLU -> FC
        """

        # CNN->ReLU->Pool
        x = self.pool(F.relu(self.conv1(x)))
        if self.apply_drop:
            x = self.drop1(x)
        
        # CNN->ReLU->Pool
        x = self.pool(F.relu(self.conv2(x)))
        if self.apply_drop:
            x = self.drop2(x)
        
        # Flatten
        x = x.view(-1, 16*5*5)
        
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.fc_drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.fc_drop2(x)
        
        # FC
        x = self.fc3(x)
        
        return x

class CIFAR_CNN_nopool(nn.Module):
    """
    CNN based model for CIFAR without Max-pooling
    
    Schema: CNN->CNN->ReLU --> CNN->CNN->ReLU --> Flatten -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(CIFAR_CNN_nopool, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 9)
        self.conv1b = nn.Conv2d(6, 6, 9)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.conv2b = nn.Conv2d(16, 16, 7)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        if droprate is not None:
            self.apply_drop = True

            self.drop1a = nn.Dropout2d(droprate)
            self.drop1b = nn.Dropout2d(droprate)
            self.drop2a = nn.Dropout2d(droprate)
            self.drop2b = nn.Dropout2d(droprate)
            self.fc_drop1 = nn.Dropout(droprate)
            self.fc_drop2 = nn.Dropout(droprate)
        else:
            self.apply_drop = False

    def forward(self, x):
        """
        Forward pass. CNN->CNN->ReLU --> CNN->CNN->ReLU --> Flatten -> FC->ReLU -> FC->ReLU -> FC
        """

        # CNN
        x = self.conv1(x)
        if self.apply_drop:
            x = self.drop1a(x)
        # CNN->ReLU
        x = self.conv1b(x)
        x = F.relu(x)
        if self.apply_drop:
            x = self.drop1b(x)

        # CNN
        x = self.conv2(x)
        if self.apply_drop:
            x = self.drop2a(x)
        # CNN->ReLU
        x = self.conv2b(x)
        x = F.relu(x)
        if self.apply_drop:
            x = self.drop2b(x)
        
        # Flatten
        x = x.view(-1, 16*4*4)
        
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.fc_drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.fc_drop2(x)
        
        # FC
        x = self.fc3(x)
        
        return x  
    
class CIFAR_DNN(nn.Module):
    """
    MLP based model for CIFAR 
    
    Schema: Flatten -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(CIFAR_DNN, self).__init__()

        self.fc1   = nn.Linear(32*32*3, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, 10)

        if droprate is not None:
            self.apply_drop = True

            self.fc_drop1 = nn.Dropout(droprate)
            self.fc_drop2 = nn.Dropout(droprate)
            self.fc_drop3 = nn.Dropout(droprate)

        else:
            self.apply_drop = False

        

    def forward(self, x):
        """
        Forward pass. Flatten -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
        """
        # Flatten
        x = x.view(-1, 32*32*3)

        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.fc_drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.fc_drop2(x)
        
        # FC->ReLU
        x = F.relu(self.fc3(x))
        if self.apply_drop:
            x = self.fc_drop3(x)
        
        # FC
        x = self.fc4(x)

        return x



### MNIST/Fashion MNIST Models ### 
class MNIST_CNN(nn.Module):
    """
    CNN based model for MNIST/Fashion-MNIST
    
    Schema: CNN->ReLU->Pool --> CNN->ReLU->Pool --> Flatten -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(MNIST_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        if droprate is not None:
            self.apply_drop = True
            self.drop1 = nn.Dropout2d(droprate)
            self.drop2 = nn.Dropout2d(droprate)
            self.fc_drop1 = nn.Dropout(droprate)
            self.fc_drop2 = nn.Dropout(droprate)
        else:
            self.apply_drop = False

    def forward(self, x):
        """
        Forward pass. CNN->ReLU->Pool --> CNN->ReLU->Pool --> Flatten -> FC->ReLU -> FC->ReLU -> FC
        """
        # CNN->ReLU->Pool
        x = self.pool(F.relu(self.conv1(x)))
        if self.apply_drop:
            x = self.drop1(x)
        
        # CNN->ReLU->Pool
        x = self.pool(F.relu(self.conv2(x)))
        if self.apply_drop:
            x = self.drop2(x)

        # Flatten
        x = x.view(-1, 256)
        
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.fc_drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.fc_drop2(x)
        
        # FC
        x = self.fc3(x)

        return x

class MNIST_CNN_nopool(nn.Module):
    """
    CNN based model for MNIST/Fashion-MNIST without Max-pooling
    
    Schema: CNN->CNN->ReLU --> CNN->CNN->ReLU --> Flatten -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(MNIST_CNN_nopool, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 9)
        self.conv1b = nn.Conv2d(6, 6, 9)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2b = nn.Conv2d(16, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
        if droprate is not None:
            self.apply_drop = True

            self.drop1a = nn.Dropout2d(droprate)
            self.drop1b = nn.Dropout2d(droprate)
            self.drop2a = nn.Dropout2d(droprate)
            self.drop2b = nn.Dropout2d(droprate)
            self.fc_drop1 = nn.Dropout(droprate)
            self.fc_drop2 = nn.Dropout(droprate)
        else:
            self.apply_drop = False


    def forward(self, x):
        """
        Forward Pass. CNN->CNN->ReLU --> CNN->CNN->ReLU --> Flatten -> FC->ReLU -> FC->ReLU -> FC
        """

        # CNN
        x = self.conv1(x)
        if self.apply_drop:
            x = self.drop1a(x)
        
        # CNN->ReLU
        x = self.conv1b(x)
        x = F.relu(x)
        if self.apply_drop:
            x = self.drop1b(x)

        # CNN->ReLU
        x = self.conv2(x)
        if self.apply_drop:
            x = self.drop2a(x)
        # CNN->ReLU
        x = self.conv2b(x)
        x = F.relu(x)
        if self.apply_drop:
            x = self.drop2b(x)
        
        x = x.view(-1, 256)

        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.fc_drop1(x)
        
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.fc_drop2(x)
        
        x = self.fc3(x)
        
        return x

class MNIST_DNN(nn.Module):
    """
    MLP based model for MNIST/Fashion-MNIST 
    
    Schema: Flatten -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout rate as input (set to `None` if not being used)
    """
    def __init__(self, droprate=None):
        """
        Constructor
        `droprate`: Dropout rate. If set to None, droput is not used
        """
        super(MNIST_DNN, self).__init__()

        self.fc1   = nn.Linear(28*28, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, 10)
        
        if droprate is not None:
            self.apply_drop = True

            self.drop1 = nn.Dropout(droprate)
            self.drop2 = nn.Dropout(droprate)
            self.drop3 = nn.Dropout(droprate)
        else:
            self.apply_drop = False

    def forward(self, x):
        """
        Forward pass. Flatten -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
        """
        # Flatten
        x = x.view(-1, 28*28)

        # FC->ReLU 
        x = F.relu(self.fc1(x))
        if self.apply_drop:
            x = self.drop1(x)

        # FC->ReLU 
        x = F.relu(self.fc2(x))
        if self.apply_drop:
            x = self.drop2(x)

        # FC->ReLU 
        x = F.relu(self.fc3(x))
        if self.apply_drop:
            x = self.drop3(x)

        # FC
        x = self.fc4(x)
        
        return x


#########################################################################
########################### Training Utility ###########################
#########################################################################
def train_model(net, optimizer, criterion, trainloader, valloader, PATH, device):
    """
    Function to train and save the modele. It uses cross validation

    `net`: Model to be used for training
    `optimizer`: Optimizer to be used
    `criterion`: Criterion/loss function to be used
    `trainloader`: Dataloader/iterator for training data
    `testloader`: Dataloader/iterator for validation data
    `PATH`: Location where model will be save
    `device`: Devce object to indicate whether GPU or CPU is to be used
    """

    ## Helper variables
    n_max_bad = 5 # Number of maximum bad iterations allowed (where loss is higher then previous iteration consecutively)
    max_num_epoch = 100 # Maximum number of epochs
    prev_val_loss = np.Inf # Variable to save validation loss in previous epoch
    bad_counter = 0 # Variable to count how many bad iterations have been encountered
    val_loss = 0. # Validation loss
    time_to_run = 0. # Total time take to train the model
    
    # Getting optimizer name from the object
    optim_name = optimizer.__str__().split(' ')[0]
    
    for epoch in range(max_num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0 # Variable to keep track of loss so far

        start_time = time.time() # Start time
        # Iterate over training data
        for i, data in enumerate(trainloader, 0):
            # Place data over the device
            inputs, labels = data[0].to(device), data[1].to(device)

            # Closure functions containing steps for training. Required for L-BFGS as it need to be run multiple times by it
            def closure():
                # Clear gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = net(inputs)
                # Find loss
                loss = criterion(outputs, labels)
                # Calcuate gradient
                loss.backward()
                return loss
            
            # If using LBFGS, use closure asoptimizer step (back-propagation)
            if optim_name == 'LBFGS':
                # Run forward pass and backprop
                loss = optimizer.step(closure)
            else:
                # Run forwards pass
                loss = closure()
                # Run backprop
                optimizer.step()
                
            # print statistics
            # Add currect loss to rolling sum
            running_loss += loss.item()
            if i % 500 == 0:    # Print every 500 epochs
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000)) # 2000 was left here by mistake. However it just scaledthe printed values
                running_loss = 0.0
            
            # If loss is nan, (model exploding), stop training 
            if np.isnan(loss.item()):
                print('Stopping due to nan')
                break
        
        # Update time to run
        time_to_run += (time.time() - start_time)

        ###### Vaildation ########
        val_loss_list = [] # List ot save all validation losses across batches in validation
        # Activate eveluation mode/switch-off dropouts
        net.eval()

        with torch.no_grad(): # switch off gradient calculation
            # Iterate over validation data
            for i, data in enumerate(valloader, 0):
                # Place data on device
                inputs, labels = data[0].to(device), data[1].to(device)
                # Inference/Prediction
                outputs = net(inputs)
                # Get loss
                loss = criterion(outputs, labels)
                # Add current loss to the list
                val_loss_list.append(loss.item())

        # Find mean loss accross batches
        val_loss = np.array(val_loss_list).mean()

        # Print validation accuracy
        print('Validation: Epoch: [%d] loss: %.6f' %
                      (epoch + 1, val_loss))

        # If current loss is higher than previous loss, increase counter
        curr_loss = prev_val_loss - val_loss
        if curr_loss < 0:
            bad_counter += 1
            print('Bad counter: %d' % (bad_counter))

            # If 5 bad iterations have been found, stop traing
            if bad_counter == n_max_bad:
                print('Early Stopping. Epoch: %d' % (epoch +1))
                break
        else:
            # Reset bad iteration counter
            bad_counter = 0
            # Save the model (bets so far)
            torch.save(net, PATH)
            # Update previous loss to current loss(best loss so far)
            prev_val_loss = val_loss

        # Activate training mode again for next epoch
        net.train()

    print('Finished Training')
    print(f'Total time to run: {time_to_run}s')
    
    return net

#########################################################################
################################ Metrics ################################
######################################################################### 
def training_acc(net, trainloader, PATH, device):
    """
    Function to calculate training accuracy

    `net`: Model to be used
    `trainloader`: Training data loader/iterator
    `PATH`: location where the model is saved
    `device`: Devce object to indicate whether GPU or CPU is to be used
    """
    # Load the saved model
    net = torch.load(PATH)
    # Activate eveluation mode/Switch-off dropouts
    net.eval()
    

    correct = 0 # number of correct predictions
    total = 0 # number of total predictions

    # Get predictions and calcuate accuracy
    with torch.no_grad(): # switch off gradient calculation
        # Iterate over training data
        for data in trainloader: 
            # Place data on device
            images, labels = data[0].to(device), data[1].to(device)
            # Inference/Prediction
            outputs = net(images)
            # Get predicted class as most likely prediction
            _, predicted = torch.max(outputs.data, 1)
            # aggreagte the number of classes
            total += labels.size(0)
            # aggreagte the number of correct prediction
            correct += (predicted == labels).sum().item()
    
    # Calculate accuray
    print(f'Training Accuracy: {100 * correct / total}%')
    
def get_accuracy(net, testloader, classes, PATH, device):
    """
    Function to calculate test accuracy (total and class-wise)

    `net`: Model to be used
    `testloader`: Test data loader/iterator
    `classes`: Names of the classes
    `PATH`: location where the model is saved
    `device`: Devce object to indicate whether GPU or CPU is to be used
    """
    # Load the saved model
    net = torch.load(PATH)
    # Activate eveluation mode/Switch-off dropouts
    net.eval()
    
    correct = 0 # number of correct predictions
    total = 0 # number of total predictions

    pred_labels = np.empty((0,1)) # predicted labels
    true_labels = np.empty((0,1)) # ground truth

    with torch.no_grad():# switch off gradient calculation
        # Iterate over training data
        for data in testloader:
            # Place data on device
            images, labels = data[0].to(device), data[1].to(device)
            # Inference/Prediction
            outputs = net(images)
            # Get predicted class as most likely prediction
            _, predicted = torch.max(outputs.data, 1)
            # aggreagte the number of classes
            total += labels.size(0)
            # aggreagte the number of correct prediction
            correct += (predicted == labels).sum().item()
            
            # Save predicted predicted labels and ground truth
            pred_labels = np.vstack([pred_labels, predicted.data.cpu().numpy().reshape(-1,1)])
            true_labels = np.vstack([true_labels, labels.data.cpu().numpy().reshape(-1,1)])
    
    # Calcualte test accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    

    ##### Finding accuracy for each class
    # create a data fraeme
    df = pd.DataFrame(columns = ['pred', 'true'], data=np.concatenate([pred_labels, true_labels],axis=1))
    
    # indicate which predictions were correct
    df['same'] = (df['pred'] == df['true']).astype(int)
    
    # For each class, subset data and find accuracy
    for i in range(10):
        subset = df.loc[df['true'] == i]
        
        print(f"Accuracy of {classes[i]} : {100 * subset['same'].values.sum() / subset.shape[0]}%")

        