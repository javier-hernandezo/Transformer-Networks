from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()  



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test():
    # We are not going to use backward so this will reduce memory consumption
    with torch.no_grad():
        # Switching off some parameters and layers during inference (e.g., dropout layers)
        model_classic.eval()
        model_coord.eval()
        test_loss_classic = 0
        test_loss_coord = 0
        correct_classic = 0
        correct_coord = 0
        for data, target in test_loader:
            # We send the eval data to the GPU
            data, target = data.to(device), target.to(device)
            # Inference
            output = model_classic(data)

            # sum up batch loss
            test_loss_classic += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability (class prediction)
            pred = output.max(1, keepdim=True)[1]
            # correct predictions (for computing accuracy)
            correct_classic += pred.eq(target.view_as(pred)).sum().item()

            # Inference
            output = model_coord(data)

            # sum up batch loss
            test_loss_coord += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability (class prediction)
            pred = output.max(1, keepdim=True)[1]
            # correct predictions (for computing accuracy)
            correct_coord += pred.eq(target.view_as(pred)).sum().item()
            
            
        test_loss_classic /= len(test_loader.dataset)
        print('\nClassic Model Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss_classic, correct_classic, len(test_loader.dataset),
                      100. * correct_classic / len(test_loader.dataset)))

        test_loss_coord /= len(test_loader.dataset)
        print('\nClassic Model Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss_coord, correct_coord, len(test_loader.dataset),
                      100. * correct_coord / len(test_loader.dataset)))


def visualize_stn():
    # for inference
    with torch.no_grad():
        # Get a batch of test data
        data = next(iter(test_loader))[0].to(device)
        
        input_tensor = data.cpu()
        transformed_input_tensor_classic = model_classic.stn(data).cpu()
        transformed_input_tensor_coord = model_coord.stn(data).cpu()

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
        axarr[1].set_title('Transformed Images Classic Conv')
        
        axarr[2].imshow(out_grid_coord)
        axarr[2].set_title('Transformed Images Coord Conv')


     
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    # Normalization and transposition
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
        
    # for inference
    with torch.no_grad():
        # Get a batch of test data
        data = next(iter(test_loader))[0].to(device)
        
        input_tensor = data.cpu()
        transformed_input_tensor = model_coord.stn(data).cpu()

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
test()

#Visualize the results of the transforms of a test batch of data.
visualize_stn()

plt.ioff()
plt.show()