import torch
from torch import nn


"""Based on https://github.com/Wizaron/coord-conv-pytorch implementation in PyTorch of the CoordConv paper:
"REF: An intriguing failing of convolutional neural networks and the CoordConv solution" https://arxiv.org/abs/1807.03247
 """

class AddCoordinates(object):

    """ This class adds information about the coordinates of the input images to the input tensor
    Following the process described in the CoordConv paper we normalize the coordinates to the [-1,1] range.
    """
    def __init__(self):
        self.initialized = True

    #Returns the input tensor of the images with the additional channels containing coords information.
    def __call__(self, image):

        #We obtain the properties of the input tensors (images)
        batch_size, channels, height, width = image.size()

        #We create the coordinate channels (normalized to the -1,1 range
        y = 2.0 * torch.arange(height).unsqueeze(1).expand(height, width) / (height - 1.0) - 1.0
        x = 2.0 * torch.arange(width).unsqueeze(0).expand(height, width) / (width - 1.0) - 1.0
        
        #We add the coordinates to form the output tensor
        coords = torch.stack((y, x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        
        return torch.cat((coords.to(image.device), image), dim=1)


class CoordConv(nn.Module):

    """ This class is designed to be used a substitute of the Conv2D layer. 
    Inputs: the same than nn.Conv2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        
        #It inherits the properties from nn.Module
        super(CoordConv, self).__init__()

        #We add the two additional input channels (x,y coords) to the Conv layer
        in_channels += 2
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.coord = AddCoordinates()

    #Method to be called for inference
    def forward(self, x):
        #The input tensor (an image) goes through the CoordConv layer
        x = self.coord(x)
        #The new tensor with the additional channels (coordinates) goes through the convolutional layer
        x = self.conv_layer(x)
        return x