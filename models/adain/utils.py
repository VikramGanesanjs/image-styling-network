import torch
import matplotlib.pyplot as plt
import numpy as np

def adjust_learning_rate(optimiser, iters, learning_rate_decay, LR):
    for param_group in optimiser.param_groups:
        param_group['lr'] = LR / (1.0 + learning_rate_decay * iters)

def concat_img(imgs, batch):
    plt.figure()
    #imgs = (imgs + 1) / 2
    imgs = imgs.movedim((0, 1, 2, 3), (0, 3, 1, 2)).detach().cpu().numpy() 
    axs = plt.imshow(np.concatenate(imgs.tolist(), axis=1))
    plt.axis('off')
    plt.savefig("../../produced-images/batch{}img.png".format(batch))
    plt.close()

def concat_img(imgs, batch):
    plt.figure()
    #imgs = (imgs + 1) / 2
    imgs = imgs.movedim((0, 1, 2, 3), (0, 3, 1, 2)).detach().cpu().numpy() 
    axs = plt.imshow(np.concatenate(imgs.tolist(), axis=1))
    plt.axis('off')
    plt.savefig("../../produced-images/batch{}img.png".format(batch))

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
    # mean = mean.view(mean.shape[0], mean.shape[1], 1, 1)
    # std = std.view(std.shape[0], std.shape[1], 1, 1)

    mean = mean.view(mean.shape[0], mean.shape[1], 1, 1)
    std = std.view(std.shape[0], std.shape[1], 1, 1)

    return (mean, std)