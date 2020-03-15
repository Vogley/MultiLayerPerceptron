from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm


# My computer does not have a GPU I can use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)  # <-- workaround for GPU testing, get rid of if cpu is being used. [May need if(device is "cuda:0"): ]

# Parameters
TRAIN_DIRECTORY_PATH = "cifar10/cifar10_train/"
TEST_DIRECTORY_PATH = "cifar10/cifar10_test/"

# Some static hyperparameters
INPUT_SIZE = 32*32*3      # Image Size
NUM_CLASSES = 10
NUM_EPOCHS = 5


# *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
#                       Part 2: TransferLearning
# *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
# Helper Function to change string labels to Integers and back
def convertToInteger(label):
    labels = {"airplane" : 0,
              "automobile" : 1,
              "bird" : 2,
              "cat" : 3,
              "deer" : 4,
              "dog" : 5,
              "frog" : 6,
              "horse" : 7,
              "ship" : 8,
              "truck" : 9,}
    
    return labels[label]


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """Initializes a dataset containing images and labels.
            Args:
                root_dir (string): Directory with all the images.
        """
        self.images = []
        for r, d, f in os.walk(root_dir):
            for file in f:
                if '.png' in file:
                    label = convertToInteger(r[r.rfind('/')+1:])
                    self.images.append([os.path.join(r, file), label])
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.images[index][0]
        image = io.imread(img_name)
        image = np.transpose(image, (2, 0, 1))
        label = self.images[index][1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define our model (3-layer MLP)
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.activation = nn.Tanh()                 # I'm using Tanh as my activation function. I tried others, but they didn't workout great.
        self.dropout = nn.Dropout(p=0.2) if dropout == True else None
        self.fc2 = nn.Linear(hidden_size, num_classes)
  
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3) # Reshaped to fit linear model
        out = self.fc1(x) 
        out = self.activation(out)
        if(self.dropout != None):
            out = self.dropout(out)
        out = self.fc2(out)
        return out

# Model used in PyTorch Example NOT USED IN THIS ASSINGMENT
'''class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x'''


def initializeDatasets():
    #Initializing Datasets
    train_dataset = CifarDataset(TRAIN_DIRECTORY_PATH)
    test_dataset = CifarDataset(TEST_DIRECTORY_PATH)

    return train_dataset, test_dataset

def trainModel(dataset, batch_size, hidden_size, dropout, optimizer, weight_decay, learning_rate):
    trainloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = MultilayerPerceptron(INPUT_SIZE, hidden_size, NUM_CLASSES, dropout).to(device)

    # I'm using Cross Entropy Loss for image classification
    criterion = nn.CrossEntropyLoss()
    
    # Choose your Optimizer.
    if optimizer == 0:
        #SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay = weight_decay)
    elif optimizer == 1:
        #SGD with momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = weight_decay)
    elif optimizer == 2:
        #Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    else:
        #RMSProp
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)

    total_epochs = tqdm(range(NUM_EPOCHS))

    #For Tracking accuracy through testing
    tracker = np.empty((0))

    model.train()           # Set the model into `training` mode, because certain operators will perform differently during training and evaluation (e.g. dropout and batch normalization)
    model = model.float()   # Model and Inputs must be of the same type
    print("Training....")
    for epoch in total_epochs:
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (sampled_batch) in enumerate(trainloader):  
            images = sampled_batch['image']
            labels = sampled_batch['label']

            # Move tensors to the configured device
            images.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images.float())     # Model and Inputs must be of the same type
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Tracking Accuracy
            tracker = np.append(tracker, 100.*correct/total)

            if (batch_idx + 1) % 10 == 0:
                total_epochs.set_description(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.3f} ({}/{})'.format(
                        epoch + 1, NUM_EPOCHS, batch_idx + 1, len(trainloader), loss.item(), 100.*correct/total, correct, total))

    print("Trained!")
    return model, tracker


def testModel(dataset, model, batch_size):
    test_dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for sampled_batch in tqdm(test_dataloader):
            images = sampled_batch['image']
            labels = sampled_batch['label']
            
            # Move tensors to the configured device
            images.to(device)
            labels.to(device)

            outputs = model(images.float())     # Model and Inputs must be of the same type
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
    return 100*correct/total


def evaluation():
    print("Training whole model...")
    # Arguments for trainModel(batch_size, hidden_size, dropout, weight_decay, learning_rate)
    
    # I don't know how to use tensorboard, and I don't have the time to figure it out
    
    hyperparameters = {
        'batch_size'    : [10, 50, 100],
        'learning_rate' : [1E-4, 1E-5, 1E-6],
        'hidden_size'   : [10000, 50000, 100000],
        'dropout'       : [False, True],
        'optimizer'     : ["SGD", "SGD+Momentum", "Adam", "RMSProp"],
        'weight_decay'  : [0, 0.01, 0.001]
    }

    # Defaults
    default = {
        'batch_size'    : 50,
        'learning_rate' : 1E-5,
        'hidden_size'   : 50000,
        'dropout'       : False,
        'optimizer'     : 1,
        'weight_decay'  : 0
    }

    # Grab datasets
    (train, test) = initializeDatasets()

    outfile = open("report.txt","w+")
    outfile.write("Report for Homework 3\n")
    outfile.write("Default Hyperparameters: " + str(default) + "\n")
    outfile.write("-----------------------------------------------------------------------------------------------------------------------\n")

    # ------------ Test Learning Rates ------------ 
    outfile.write("Testing Learning Rates\n")

    #I wanted to see a plot of the different values.
    accuracies = []
    for l in hyperparameters["learning_rate"]:
        print("Testing learning_rate: " + str(l) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, default['batch_size'], default['hidden_size'], default['dropout'], default['optimizer'], default['weight_decay'], l)
        accuracy = testModel(test, trainedModel, default['batch_size'])
        outfile.write("Learning Rate of " + str(l) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)

    # Show the plot for learning rates
    plt.plot(accuracies[0], 'b', label='n = 0.0001')
    plt.plot(accuracies[1], 'r', label='n = 0.00001')
    plt.plot(accuracies[2], 'g', label='n = 0.000001')
    plt.legend(loc="upper left")
    plt.savefig('learning_rates.png')
    plt.show()
    plt.clf()


    # ------------ Test Batch Sizes ------------ 
    outfile.write("Testing Batch Sizes\n")

    accuracies = []
    for b in hyperparameters["batch_size"]:
        print("Testing batch_size: " + str(b) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, b, default['hidden_size'], default['dropout'], default['optimizer'], default['weight_decay'], default['learning_rate'])
        accuracy = testModel(test, trainedModel, b)
        outfile.write("Batch_Size of " + str(b) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)
    
    # Show the plot for Batch Sizes
    plt.plot(accuracies[0], 'b', label='b = 10')
    plt.plot(accuracies[1], 'r', label='b = 50')
    plt.plot(accuracies[2], 'g', label='b = 100')
    plt.legend(loc="upper left")
    plt.savefig('batch_sizes.png')
    plt.show()
    plt.clf()

    # ------------ Test Hidden Sizes ------------ 
    # This is the most important parameter I have found. Needs to be complex to have a high training accuracy.
    outfile.write("Testing Hidden Sizes\n")

    accuracies = []
    for h in hyperparameters["hidden_size"]:
        print("Testing hidden_size: " + str(h) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, default['batch_size'], h, default['dropout'], default['optimizer'], default['weight_decay'], default['learning_rate'])
        accuracy = testModel(test, trainedModel, default['batch_size'])
        outfile.write("Hidden_Size of " + str(h) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)
    
    # Show the plot for Hidden Sizes
    plt.plot(accuracies[0], 'b', label='h = 10000')
    plt.plot(accuracies[1], 'r', label='h = 50000')
    plt.plot(accuracies[2], 'g', label='h = 100000')
    plt.legend(loc="upper left")
    plt.savefig('hidden_sizes.png')
    plt.show()
    plt.clf()

    # ------------ Test Dropout ------------ 
    outfile.write("Testing Dropout\n")

    accuracies = []
    for d in hyperparameters["dropout"]:
        print("Testing dropout: " + str(d) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, default['batch_size'], default['hidden_size'], d, default['optimizer'], default['weight_decay'], default['learning_rate'])
        accuracy = testModel(test, trainedModel, default['batch_size'])
        outfile.write("Dropout of " + str(d) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)
    
    # Show the plot for Hidden Sizes
    plt.plot(accuracies[0], 'b', label='No Dropout')
    plt.plot(accuracies[1], 'r', label='Dropout')
    plt.legend(loc="upper left")
    plt.savefig('dropout.png')
    plt.show()
    plt.clf()

    # ------------ Test Optimizer ------------ 
    outfile.write("Testing Optimizers\n")

    accuracies = []
    for o in range(len(hyperparameters["optimizer"])):
        print("Testing optimizer: " + str(hyperparameters["optimizer"][o]) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, default['batch_size'], default['hidden_size'], default['dropout'], o, default['weight_decay'], default['learning_rate'])
        accuracy = testModel(test, trainedModel, default['batch_size'])
        outfile.write("Optimizer of " + str(hyperparameters["optimizer"][o]) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)
    
    # Show the plot for Hidden Sizes
    plt.plot(accuracies[0], 'b', label='SGD')
    plt.plot(accuracies[1], 'r', label='SGD + Momentum')
    plt.plot(accuracies[2], 'g', label='Adam')
    plt.plot(accuracies[3], 'y', label='RMSProp')
    plt.legend(loc="upper left")
    plt.savefig('optimizers.png')
    plt.show()
    plt.clf()

    # ------------ Test Weight Decay ------------ 
    outfile.write("Testing Weight Decay\n")

    accuracies = []
    for w in hyperparameters["weight_decay"]:
        print("Testing weight_decay: " + str(w) + "\n")
        (trainedModel, accuracy_tracker) = trainModel(train, default['batch_size'], default['hidden_size'], default['dropout'], default['optimizer'], w, default['learning_rate'])
        accuracy = testModel(test, trainedModel, default['batch_size'])
        outfile.write("Weight Decay of " + str(w) + " scored an accuracy of " + str(accuracy) + "\n")
        accuracies.append(accuracy_tracker)
    
    # Show the plot for Hidden Sizes
    plt.plot(accuracies[0], 'b', label='w = 0')
    plt.plot(accuracies[1], 'r', label='w = 0.01')
    plt.plot(accuracies[2], 'g', label='w = 0.001')
    plt.legend(loc="upper left")
    plt.savefig('weight_decay.png')
    plt.show()
    plt.clf()

    outfile.close()


# *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
#                       Part 3: TransferLearning
# *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def transferLearning(feature_extract):
    print("Transfer learning using MobileNet V2")
    print("Feature Extraction: " + str(feature_extract))

    # Set up Pretrained Model
    mobilenet = models.mobilenet_v2(pretrained=True, progress=True) # Load in model
    set_parameter_requires_grad(mobilenet, True) # Only Finetune last layer.
    num_ftrs = mobilenet.classifier[1].in_features
    mobilenet.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    # Model Initialized

    # Grab datasets
    (train, test) = initializeDatasets()
    trainloader = torch.utils.data.DataLoader(train,
                                               batch_size=50,
                                               shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(test,
                                               batch_size=50,
                                               shuffle=True)

    # Send the model to GPU
    model_ft = mobilenet.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

    # Setup the loss fxn
    criterion_ft = nn.CrossEntropyLoss()

    # ******* Train Model *******
    total_epochs = tqdm(range(NUM_EPOCHS))

    #For Tracking accuracy through testing
    tracker = np.empty((0))

    model_ft.train()        # Set the model into `training` mode, because certain operators will perform differently during training and evaluation (e.g. dropout and batch normalization)
    model_ft = model_ft.float()   # Model and Inputs must be of the same type
    print("Training....")
    for epoch in total_epochs:
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (sampled_batch) in enumerate(trainloader):  
            images = sampled_batch['image']
            labels = sampled_batch['label']

            # Move tensors to the configured device
            images.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # Forward pass
            outputs = model_ft(images.float())     # Model and Inputs must be of the same type
            loss = criterion_ft(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer_ft.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Tracking Accuracy
            tracker = np.append(tracker, 100.*correct/total)

            if (batch_idx + 1) % 10 == 0:
                total_epochs.set_description(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.3f} ({}/{})'.format(
                        epoch + 1, NUM_EPOCHS, batch_idx + 1, len(trainloader), loss.item(), 100.*correct/total, correct, total))

    print("Trained!")

    # **********  Test the model ***********
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model_ft.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for sampled_batch in tqdm(test_dataloader):
            images = sampled_batch['image']
            labels = sampled_batch['label']
            
            # Move tensors to the configured device
            images.to(device)
            labels.to(device)

            outputs = model_ft(images.float())     # Model and Inputs must be of the same type
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
    test_accuracy = 100*correct/total

    # Report on Findings
    plt.plot(tracker, 'b', label='Feature Extraction: ' + str(feature_extract))
    plt.legend(loc="upper left")
    plt.savefig('transfer_learning_'+str(feature_extract)+'.png')
    plt.show()
    plt.clf()



if __name__ == "__main__":
    # Part 2 is evaluation. Part 3 is TransferLearning.
    
    evaluation()
    transferLearning(feature_extract = True)
    transferLearning(feature_extract = False)