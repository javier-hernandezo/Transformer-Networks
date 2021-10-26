import torch
from torch import nn


"""Based on https://github.com/Wizaron/coord-conv-pytorch and implementation in PyTorch of the CoordConv paper """

class AddCoordinates(object):

    """ This class adds the information about the coordinates of the input images
    Following the process described in the reference paper [REF] we normalize the coordinates to the [-1,1] range.
    """
    def __init__(self):
        self.initialized = True

    #Returns the input images with additional channels containing coords information.
    def __call__(self, image):
    
        batch_size, channels, height, width = image.size()

        y = 2.0 * torch.arange(height).unsqueeze(1).expand(height, width) / (height - 1.0) - 1.0
        x = 2.0 * torch.arange(width).unsqueeze(0).expand(height, width) / (width - 1.0) - 1.0
        
        coords = torch.stack((y, x), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        
        return torch.cat((coords.to(image.device), image), dim=1)


class CoordConv(nn.Module):

    """ This class is a substitution for the Conv2D layer in PyTorch. 
    Input: same than nn.Conv2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        
        #Inherits properties from nn.Module
        super(CoordConv, self).__init__()

        #We add the two additional input channels (x,y coords) to the Conv layer
        in_channels += 2
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.coord = AddCoordinates()

    def forward(self, x):
    
        x = self.coord(x)
        x = self.conv_layer(x)
        return x