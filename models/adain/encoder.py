import torch
import torchvision

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        # Split the model into each ReLu layer so that we can compute the loss and train the whole model
        # There are 21 different layers in the model, but the list is not cut off there, so we need to splice it
        layers = list(model.features.children())[:21]
        
        # the eval sets our sequential model into evaluation mode, so that we can split up the layers
        f = torch.nn.Sequential(*layers).eval()

        #splitting up the layers of f
        self.relu_1_1 = torch.nn.Sequential(*f[:2])
        self.relu_2_1 = torch.nn.Sequential(*f[2:5], *f[5:7])
        self.relu3_1 = torch.nn.Sequential(*f[7:10],*f[10:12])
        self.relu4_1 = torch.nn.Sequential(*f[12:14],*f[14:16],f[16:19],*f[19:21])


    def forward(self, x):
        #propagate through each of our sequential layers, saving each to compute the losses later
        output1 = self.relu_1_1(x)
        output2 = self.relu_2_1(output1)
        output3 = self.relu3_1(output2)
        output4 = self.relu4_1(output3)

        return output1, output2, output3, output4