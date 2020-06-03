__author__ = "Vishnu Dutt Sharma"
__description__ = "Utilities for experiments on classifier networks"

import time

import random
import numpy as np

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
sys.path.insert(0,'KFAC_Pytorch/')
from KFAC_Pytorch import optimizers

# Importing function to find MSE
from sklearn.metrics import mean_squared_error

###################################################################
######################## Data Utilities ###########################
###################################################################
class Dataset():
    """
    Dataset utility for dataset class
    """
    def __init__(self, name="boston_housing", batch_size=None):
        """
        Constructor to create Boston Housing dataset
        
        `dataset`: Name of the dataset. Options: 'boston_housing', 'breast_cancer'
        `batchsize`: Batch Size. If None is given as input, whole dataset is use (no mini-batch)
        """

        # Import utilities to split dataset
        from sklearn.model_selection import train_test_split
        
        data_obj = None
        
        if name == "boston_housing":
            from sklearn.datasets import load_boston
            data_obj = load_boston()
            
        elif name == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            data_obj = load_breast_cancer()
        
        else:
            print('Invalid dataset name. Allowed names: boston_housing, breast_cancer')
        
        print(f'Loading {name} dataset')

        # Split  data into train, validation+test dataset in ratio 6:4
        X_train, X_valtest, y_train, y_valtest = train_test_split(data_obj.data, data_obj.target.reshape(-1,1), test_size=0.6, random_state=123)
        # Split validation+test into separate datasets into 1:1 ratio
        X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=123)

        ######## Data normalization ##########
        # Find mean and standard deviation for traing data and targets
        mean_X, std_X = X_train.mean(0), X_train.std(0)
        mean_Y, std_Y = y_train.mean(0), y_train.std(0)

        # Normalize all datasets splits
        X_train = (X_train - mean_X) / std_X
        X_val   = (X_val   - mean_X) / std_X
        X_test  = (X_test  - mean_X) / std_X

        # Normalize only the training targets. 
        # Predictions will be denormalzied during evaludation
        y_train = (y_train - mean_Y) / std_Y
        # y_val   = (y_val   - mean_Y) / std_Y
        # y_test  = (y_test  - mean_Y) / std_Y
        
        # Save the normalizing parameters to dataset
        self.mean_X = mean_X
        self.std_X = std_X
        self.mean_Y = mean_Y
        self.std_Y = std_Y
        
        # If no batchsize is given, set it to dataset size (no mini-batch)
        if batch_size == None:
            batch_size = X_train.shape[0]
        
        print(f'BatchSize(training) is set to {batch_size}')
        #print(f'Scaling paramaters:')
        #print(f'Mean(X):{self.mean_X}, Std(X):{std_X}')
        #print(f'Mean(Y):{self.mean_Y}, Std(Y):{std_Y}')
        
        # Create dataloaders/iterators
        train = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=True)

        val = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        self.valloader = torch.utils.data.DataLoader(val, batch_size = 512, shuffle=False)

        test = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        self.testloader = torch.utils.data.DataLoader(test, batch_size = 512, shuffle=False)


#########################################################################
######################### Network Architectures #########################
#########################################################################

class DNN3(nn.Module):
    """
    MLP based model with 3 layers 
    
    Schema: FC->ReLU -> FC->ReLU -> FC-> FC
    FC: Fully connnected
    
    Take dropout indicator as input (set to True if dropout is to used)
    """
    def __init__(self, in_feat_num, dropstate=False):
        """
        Constructor
        `in_feat_num`: Number of features in input
        `dropstate`: Indicator to show if dropout is to be used
        """
        super(DNN3, self).__init__()
        self.dropstate = dropstate
        
        self.fc1   = nn.Linear(in_feat_num, 16)
        self.fc2   = nn.Linear(16, 8)
        self.fc3   = nn.Linear(8, 1)
        
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forwards Pass. FC->ReLU -> FC->ReLU -> FC-> FC
        """
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.dropstate:
            x = self.drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.dropstate:
            x = self.drop2(x)
        
        # FC 
        x = self.fc3(x)
        
        return x
    
class DNN4(nn.Module):
    """
    MLP based model with 4 layers 
    
    Schema: FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout indicator as input (set to True if dropout is to used)
    """
    def __init__(self, in_feat_num, dropstate=False):
        """
        Constructor
        `in_feat_num`: Number of features in input
        `dropstate`: Indicator to show if dropout is to be used
        """
        super(DNN4, self).__init__()
        self.dropstate = dropstate
        
        self.fc1   = nn.Linear(in_feat_num, 32)
        self.fc2   = nn.Linear(32, 16)
        self.fc3   = nn.Linear(16, 8)
        self.fc4   = nn.Linear(8, 1)
        
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forwards Pass. FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
        """
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.dropstate:
            x = self.drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.dropstate:
            x = self.drop2(x)
        
        # FC->ReLU
        x = F.relu(self.fc3(x))
        if self.dropstate:
            x = self.drop3(x)
        
        # FC
        x = self.fc4(x)
        
        return x
    
class DNN5(nn.Module):
    """
    MLP based model with 5 layers 
    
    Schema: FC->ReLU -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
    FC: Fully connnected
    
    Take dropout indicator as input (set to True if dropout is to used)
    """
    def __init__(self, in_feat_num, dropstate=False):
        """
        Constructor
        `in_feat_num`: Number of features in input
        `dropstate`: Indicator to show if dropout is to be used
        """
        super(DNN5, self).__init__()
        self.dropstate = dropstate
        
        self.fc1   = nn.Linear(in_feat_num, 64)
        self.fc2   = nn.Linear(64, 32)
        self.fc3   = nn.Linear(32, 16)
        self.fc4   = nn.Linear(16, 8)
        self.fc5   = nn.Linear(8, 1)
        
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forwards Pass. FC->ReLU -> FC->ReLU -> FC->ReLU -> FC->ReLU -> FC
        """
        # FC->ReLU
        x = F.relu(self.fc1(x))
        if self.dropstate:
            x = self.drop1(x)
        
        # FC->ReLU
        x = F.relu(self.fc2(x))
        if self.dropstate:
            x = self.drop2(x)
        
        # FC->ReLU
        x = F.relu(self.fc3(x))
        if self.dropstate:
            x = self.drop3(x)
        
        # FC->ReLU
        x = F.relu(self.fc4(x))
        if self.dropstate:
            x = self.drop4(x)
        
        # FC
        x = self.fc5(x)
        
        return x



#########################################################################
########################### Training Utility ###########################
#########################################################################
def train_model(net, optimizer, criterion, dataset, PATH, device):
    """
    Function to train and save the modele. It uses cross validation

    `net`: Model to be used for training
    `optimizer`: Optimizer to be used
    `criterion`: Criterion/loss function to be used
    `dataset`: Dataset object containing dataloaders and other info
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
        for i, data in enumerate(dataset.trainloader, 0):
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
            # Add currect loss ti rolling sum
            running_loss += loss.item()
            if i % 10 == 0:    # Print every 10 epochs
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 2000)) # 2000 was left here by mistake. However it just scaledthe printed values
                running_loss = 0.0
            
            # If loss is nan, (model exploding), stop training 
            if np.isnan(loss.item()):
                print('Stopping due to nan')
                break
        
        # Update time to run
        time_to_run += (time.time() - start_time)

        ###### Validation ########
        # Activate eveluation mode/switch-off dropouts
        net.eval()

        # helper variables
        pred_arr = np.empty((0,1)) # predicted labels
        ytrue_arr = np.empty((0,1)) # ground truth labels
        with torch.no_grad(): # switch off gradient calculation
            # Iterate over validation data
            for i, data in enumerate(dataset.valloader, 0):
                # Place data on device
                inputs, labels = data[0].to(device), data[1].to(device)
                # Inference/Prediction
                outputs = net(inputs)
                # Save predicted predicted labels and ground truth
                pred_arr = np.vstack([pred_arr, outputs.data.cpu().numpy().reshape(-1,1)])
                ytrue_arr = np.vstack([ytrue_arr, labels.data.cpu().numpy().reshape(-1,1)])
        
        # Scale the predictions to original scale (denormalize)
        pred_arr = (pred_arr * dataset.std_Y) + dataset.mean_Y
        # Find MSE
        val_loss = mean_squared_error(ytrue_arr,pred_arr )

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
def training_acc(net, dataset, PATH, device):
    """
    Function to calculate training accuracy

    `net`: Model to be used
    `dataset`: Dataset object containing training data loader/iterator
    `PATH`: location where the model is saved
    `device`: Devce object to indicate whether GPU or CPU is to be used
    """
    # Load the saved model
    net = torch.load(PATH)
    # Activate eveluation mode/Switch-off dropouts
    net.eval()

    # placeholder for predicttion and ground truth
    pred_arr = np.empty((0,1))
    ytrue_arr = np.empty((0,1))

    # Get predictions and calcuate accuracy
    with torch.no_grad():# switch off gradient calculation
        # Iterate over training data
        for data in dataset.trainloader:
            # Place data on device
            images, labels = data[0].to(device), data[1].to(device)
            # Inference/Prediction
            outputs = net(images)
            # Add the prediction and ground truth numpy array to variables 
            pred_arr = np.vstack([pred_arr, outputs.data.cpu().numpy().reshape(-1,1)])
            ytrue_arr = np.vstack([ytrue_arr, labels.data.cpu().numpy().reshape(-1,1)])
    
    # Scale/denormalize the predictions
    pred_arr = (pred_arr * dataset.std_Y) + dataset.mean_Y
    # Find MSE
    print(f'Training MSE: {mean_squared_error(ytrue_arr,pred_arr )}')
    
def get_accuracy(net, dataset, PATH, device):
    """
    Function to calculate test accuracy

    `net`: Model to be used
    `dataset`: Dataset object containing test data loader/iterator
    `PATH`: location where the model is saved
    `device`: Devce object to indicate whether GPU or CPU is to be used
    """
    # Load the saved model
    net = torch.load(PATH)
    # Activate eveluation mode/Switch-off dropouts
    net.eval()

    # placeholder for predicttion and ground truth
    pred_arr = np.empty((0,1))
    ytrue_arr = np.empty((0,1))

   # Get predictions and calcuate accuracy
    with torch.no_grad():# switch off gradient calculation
        # Iterate over test data
        for data in dataset.testloader:
            # Place data on device
            images, labels = data[0].to(device), data[1].to(device)
            # Inference/Prediction
            outputs = net(images)
            # Add the prediction and ground truth numpy array to variables 
            pred_arr = np.vstack([pred_arr, outputs.data.cpu().numpy().reshape(-1,1)])
            ytrue_arr = np.vstack([ytrue_arr, labels.data.cpu().numpy().reshape(-1,1)])
    
    # Scale/denormalize the predictions
    pred_arr = (pred_arr * dataset.std_Y) + dataset.mean_Y
    # Find MSE
    print(f'MSE of the network: {mean_squared_error(ytrue_arr,pred_arr )}')


