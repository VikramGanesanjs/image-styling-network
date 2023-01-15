import torch
from utils import *


class AdaIN(torch.nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        # initialize instance normalization function
        # this is the basis of our AdaIN layer, it follows an equation similar to a z-score
        # (x - mu)/sigma
        self.instance_norm = torch.nn.InstanceNorm2d(3)
    
    # forward method for our layer
    # x would be the content input and y would be the style input
    # both x and y are tensors
    def forward(self, x, y):
        # size is shaped (N, num_channels, Height, Width)
        x_size = x.size()
        
        # we do not need these since they will be calculated by the instance normalization function
        #x_mean, x_std = mean_and_std_of_image(x)
        y_mean, y_std = mean_and_std_of_image(y)

        x_norm = self.instance_norm(x)

        
        print(x_norm.size())
        # expand size of tensors so that there are no shape errors when performing AdaIN operation
        # if not self.training:
        #     x_norm = x_norm.view(*x_norm.shape, 1)

        x_size = x_norm.size()
        print(x_size)
        return y_std.expand(x_size) * x_norm + y_mean.expand(x_size)



