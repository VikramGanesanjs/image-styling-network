import torch

# takes in image tensor x as input
def mean_and_std_of_image(x):
    x_size = x.size()
    # turn x into the shape of (batch_size, num_channels, height*width)
    x = x.view(x.shape[0], x.shape[1], -1)
    #calculate the mean of the second dimension, H*W
    mean = x.mean(dim=2)
    std = x.var(dim=2).sqrt()
    #reshape mean and std to size (batch_size, num_channels, 1, 1)
    #because mean and std are sort of a scalar quantity the last two dimensions are both 1
    mean = mean.view(mean.shape[0], mean.shape[1], 1, 1)
    std = std.view(std.shape[0], std.shape[1], 1, 1)

    return (mean, std)

class AdaIN(torch.nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        # initialize instance normalization function
        # this is the basis of our AdaIN layer, it follows an equation similar to a z-score
        # (x - mu)/sigma
        self.instance_norm = torch.nn.InstanceNorm2D(3)
    
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
        
        # expand size of tensors so that there are no shape errors when performing AdaIN operation
        return y_std.expand(x_size) * x_norm + y_mean.expand(x_size)



