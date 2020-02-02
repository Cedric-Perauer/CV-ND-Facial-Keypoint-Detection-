## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #input size : 224x224x1
        self.conv1 = nn.Conv2d(1, 32, 5)
        #input size : 110x110x32
        self.conv2 = nn.Conv2d(32,64, 5)
        #input size : 53x53x64
        self.conv3 = nn.Conv2d(64,128, 5)
        #input size : 24x24x128
        self.conv4 = nn.Conv2d(128,128, 5)
        #conv out : 10x10x256
        self.pool = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(10*10*128,3000)
        self.bn_dense1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(3000,500)
        self.bn_dense2 = nn.BatchNorm1d(136) 
        self.fc3 = nn.Linear(500,136)
        
        self.drop = nn.Dropout(0.4)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))        
        x = self.pool(F.relu(self.bn_conv1(self.conv2(x))))        
        x = self.pool(F.relu(self.bn_conv2(self.conv3(x))))        
        x = self.pool2(F.relu(self.bn_conv3(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.bn_dense1(self.fc2(x)))
        x = self.drop(x)
        x = F.relu(self.bn_dense2(self.fc3(x)))
   
        # a modified x, having gone through all the layers of your model, should be returned
        return x
