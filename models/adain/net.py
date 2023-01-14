import torch
import torch.nn as nn
from adain import AdaIN
from utils import *

class StyleTransfer(nn.Module):
    def __init__(self, encoder, decoder):
        super(StyleTransfer, self).__init__()
        layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*layers[18:31])  # relu3_1 -> relu4_1]
        self.relus = [self.enc_1, self.enc_2, self.enc_3, self.enc_4]
        self.decoder = decoder
        self.mse = nn.MSELoss()
        self.adain = AdaIN()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_save(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
        
    def encode(self, input):
        res = input
        for layer in self.relus:
            res = layer(res)
        return res
    
    def forward(self, content, style):
        encoded_style = self.encode_with_save(style)
        encoded_content = self.encode(content)

        t = self.adain(encoded_content, encoded_style[-1])

        g_t = self.decoder(t)

        g_t_encoding = self.encode_with_save(g_t)

        s_loss = self.style_loss(g_t_encoding, encoded_style)
        c_loss = self.content_loss(g_t_encoding[-1], t)

        return g_t, s_loss, c_loss


    def style_loss(self, encoded_image, encoded_style):
        MSE = torch.nn.MSELoss()
        initial_mean_image, initial_std_image = mean_and_std_of_image(encoded_image[0])
        initial_mean_style, initial_std_style = mean_and_std_of_image(encoded_style[0])
        loss = MSE(initial_mean_image, initial_mean_style) + MSE(initial_std_image, initial_std_style)
        for i in range(1, 4, 1):
            mean_image, std_image = mean_and_std_of_image(encoded_image[i])
            mean_style, std_style = mean_and_std_of_image(encoded_style[i])
            loss += MSE(mean_image, mean_style) + MSE(std_image, std_style)
        return loss


    def content_loss(self, encoded_image, style_content_combined):
        MSE = torch.nn.MSELoss()
        return MSE(encoded_image, style_content_combined)





        
