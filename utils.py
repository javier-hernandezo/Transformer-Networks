import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def train(train_loader, model, optimizer, device, epoch):
    # Switching on some parameters and layers during training (e.g., dropout layers)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We send the train data to the GPU
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # We use negative log likelihood loss
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Function for testing both models over the same test data
def test(test_loader, device, model1, model2=[]):

    if bool(model2)==True:
        # We are not going to use backward so this will reduce memory consumption
        with torch.no_grad():
            # Switching off some parameters and layers during inference (e.g., dropout layers)
            model1.eval()
            model2.eval()
            test_loss_classic = 0
            test_loss_coord = 0
            correct_classic = 0
            correct_coord = 0
            for data, target in test_loader:
                # We send the eval data to the GPU
                data, target = data.to(device), target.to(device)
                # Inference
                output = model1(data)

                # sum up batch loss
                test_loss_classic += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability (class prediction)
                pred = output.max(1, keepdim=True)[1]
                # correct predictions (for computing accuracy)
                correct_classic += pred.eq(target.view_as(pred)).sum().item()

                # Inference
                output = model2(data)

                # sum up batch loss
                test_loss_coord += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability (class prediction)
                pred = output.max(1, keepdim=True)[1]
                # correct predictions (for computing accuracy)
                correct_coord += pred.eq(target.view_as(pred)).sum().item()
                
            #Print cost and accuracy over the test data    
            test_loss_classic /= len(test_loader.dataset)
            print('\nConv Model Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss_classic, correct_classic, len(test_loader.dataset),
                          100. * correct_classic / len(test_loader.dataset)))

            test_loss_coord /= len(test_loader.dataset)
            print('\nCoordConv Model Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss_coord, correct_coord, len(test_loader.dataset),
                          100. * correct_coord / len(test_loader.dataset)))
                          
    else:
        with torch.no_grad():
        # Switching off some parameters and layers during inference (e.g., dropout layers)
            model1.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                # We send the eval data to the GPU
                data, target = data.to(device), target.to(device)
                # Inference
                output = model1(data)

                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability (class prediction)
                pred = output.max(1, keepdim=True)[1]
                # correct predictions (for computing accuracy)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))

# Function for visualizing the results of the affine transformations of the LTNs.
def visualize_stn(test_loader, device, model1, model2=[]):
    
    plt.ion()
    # for inference
    if bool(model2)==True:
        with torch.no_grad():
            # Get a batch of test data
            data = next(iter(test_loader))[0].to(device)
            
            input_tensor = data.cpu()
            transformed_input_tensor_classic = model1.stn(data).cpu()
            transformed_input_tensor_coord = model2.stn(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid_classic = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor_classic))
                
            out_grid_coord = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor_coord))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid_classic)
            axarr[1].set_title('Transformed Images STN Conv')
            
            axarr[2].imshow(out_grid_coord)
            axarr[2].set_title('Transformed Images STN CoordConv')
        
    else:
        with torch.no_grad():
            # Get a batch of test data
            data = next(iter(test_loader))[0].to(device)
            
            input_tensor = data.cpu()
            transformed_input_tensor = model1.stn(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
    plt.ioff()
    plt.show()

# Function for convertir a Tensor to a numpy image     
def convert_image_np(inp):
    # Normalization and transposition
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
    
         
        
