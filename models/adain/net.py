import torch
from adain import AdaIN, mean_and_std_of_image
from encoder import Encoder
from decoder import Decoder

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
COLOR_CHANNELS = 3
LAMBDA = 1

class StyleTransfer(torch.nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        self.adain = AdaIN()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, content, style):
        c_output1, c_output2, c_output3, result_image = self.encoder(content)
        s_output1, s_output2, s_output3, result_style = self.encoder(style)

        encoded_content = [c_output1, c_output2, c_output3, result_image]

        encoded_style = [s_output1, s_output2, s_output3, result_style]

        style_content_combined = self.adain(result_image, result_style)
        style_content_combined = LAMBDA * style_content_combined + (1 - LAMBDA) * result_image

        final_image = self.decoder(style_content_combined)

        f_output1, f_output2, f_output3, result_f= self.encoder(final_image)

        encoded_image = [f_output1, f_output2, f_output3, result_f]

        return style_content_combined, encoded_style, encoded_image, final_image


def style_loss(encoded_image, encoded_style):
    MSE = torch.nn.MSELoss()
    initial_mean_image, initial_std_image = mean_and_std_of_image(encoded_image[0])
    initial_mean_style, initial_std_style = mean_and_std_of_image(encoded_style[0])
    loss = MSE(initial_mean_image, initial_mean_style) + MSE(initial_std_image, initial_std_style)
    for i in range(1, 4, 1):
        mean_image, std_image = mean_and_std_of_image(encoded_image[i])
        mean_style, std_style = mean_and_std_of_image(encoded_style[i])
        loss += MSE(mean_image, mean_style) + MSE(std_image, std_style)
    return loss


def content_loss(encoded_image, style_content_combined):
    MSE = torch.nn.MSELoss()
    return MSE(encoded_image, style_content_combined)





        
