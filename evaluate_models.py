from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import test, visualize_stn, convert_image_np
from models import Net, coordConvNet


"""
Script for the evaluation of the STN and STN + CoordConv models trained on MNIST. 

We evaluate and compare a CNN with STN, and the same network but adding 
CoordConv layers to the LTN of the STN to improve the accuracy of the affine transformations.

**Author**: `Javier Hernandez-Ortega <https://github.com/uam-biometrics>`_
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             

#Load trained models
model_classic = torch.load('stn_classic.pt')
model_coord = torch.load('stn_coordconv.pt')

# Load test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)


#Evaluate both methods on the same test data
test(test_loader, device, model_classic,model_coord)

#Visualize the results of the transforms of a test batch of data.
visualize_stn(test_loader, device, model_classic,model_coord)

