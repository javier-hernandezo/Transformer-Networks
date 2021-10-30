from __future__ import division,print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from coord_conv import AddCoordinates, CoordConv
import pdb
from six.moves import urllib
from models import coordConvNet
from utils import train, test, visualize_stn, convert_image_np


"""
Spatial Transformer Networks with CoordConv layers
=====================================
**Author**: `Javier Hernandez-Ortega <https://github.com/uam-biometrics>`_

STN part is based on the paper: "Spatial Transformer Networks" <https://arxiv.org/abs/1506.02025>`


Code to augment a STN using CoordConv layers aiming to solve some limitations of classic Conv layers 
by adding information about the coordinates of the input images to the input tensor. 
The CoordConv layer is designed to be used a substitute of the regular Conv2D layer.

You can read more about the CoordConv layers in the paper: "An intriguing failing of convolutional neural networks and the coordconv solution" <https://arxiv.org/pdf/1807.03247.pdf>
"""

"""Loading the data
----------------
Here we experiment with the classic MNIST dataset and we use a
standard convolutional network augmented with a spatial transformer
network and CoordConv layers.
"""
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0) #eran 4 num_workers
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)


model = coordConvNet().to(device)

"""Training the model"""
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1, 20 + 1):
# for epoch in range(1, 3):
    train(train_loader, model, optimizer, device, epoch)
    test(test_loader, device, model)
    
torch.save(model,'stn_coordconv.pt')

# Visualize the STN transformation on some input batch
visualize_stn(test_loader, device, model)

