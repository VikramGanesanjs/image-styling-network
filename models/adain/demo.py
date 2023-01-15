import gradio as gr
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from decoder import decoder as Decoder
from encoder import encoder as Encoder
from net import StyleTransfer
from PIL import Image

encoder = Encoder
decoder = Decoder

encoder.load_state_dict(torch.load("./vgg_normalised.pth"))
encoder = nn.Sequential(*list(encoder.children())[:31])
decoder.load_state_dict(torch.load("../saved-models/decoder_iter_1000.pth.tar"))


net = StyleTransfer(encoder, decoder)

net.eval()

def train_transform():
        transform_list = [
            transforms.Resize(size=(512, 512)),
            # transforms.CenterCrop(256),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

def cleanup(input, style):
    transform = train_transform()
    input_img = transform(Image.fromarray(input))
    style_img = transform(Image.fromarray(style))
    input_img = input_img.view(1, *input_img.shape)
    style_img = style_img.view(1, *style_img.shape)
    final_image_tensor = net(input_img, style_img)
    final_image_tensor = final_image_tensor.squeeze()
    to_pil = transforms.ToPILImage()
    image = to_pil(final_image_tensor)
    return image

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=cleanup, inputs=[gr.Image(shape=(224, 224)),gr.Image(shape=(224,224))],outputs="image")
demo.launch()