from net import StyleTransfer, content_loss, style_loss
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.multiprocessing
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')


mps_device = torch.device("mps")

# Define training constants, learning rate, and optimizer
LR = 1e-4
epochs = 1
style_transfer_network = StyleTransfer()
style_transfer_network.to(mps_device)
optimizer = torch.optim.Adam(style_transfer_network.decoder.parameters(), lr=LR)
BATCH_SIZE = 16


# Helper image show method

def concat_img(imgs):
    plt.figure()
    #imgs = (imgs + 1) / 2
    imgs = imgs.movedim((0, 1, 2, 3), (0, 3, 1, 2)).detach().cpu().numpy() 
    axs = plt.imshow(np.concatenate(imgs.tolist(), axis=1))
    plt.axis('off')
    plt.show()


#import datasets

# Set the path to the dataset directory
content_dataset_dir = '../../content-dataset/images/images'
style_dataset_dir = '../../style-dataset/images/images'

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
content_dataset = torchvision.datasets.ImageFolder(content_dataset_dir, transform=transform)
style_dataset = torchvision.datasets.ImageFolder(style_dataset_dir, transform=transform)
# Create a data loader to iterate over the dataset
content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
style_data_loader = torch.utils.data.DataLoader(style_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)





decoder = style_transfer_network.decoder
encoder = style_transfer_network.encoder

def train():
    for epoch in range(epochs):
        #train the decoder initially
        decoder.train()

        for batch, (content, _) in enumerate(content_data_loader):
            style, _ = next(iter(style_data_loader))
            content = content.to(mps_device)
            style = style.to(mps_device)

            # # Reset the gradients so that they do not get too high
            decoder.zero_grad()

            # Run the content and style through the model
            style_content_combined, encoded_style, encoded_content, final_image = style_transfer_network.forward(content, style)


            # Compute losses as outlined in the paper
            c_loss = content_loss(encoded_content[3], style_content_combined)
            s_loss = style_loss(encoded_content, encoded_style)

            # Total loss weights style_loss doubly
            total_loss = c_loss + 2 * s_loss

            # Backpropagate through the network
            total_loss.backward()
            # Change the weights according to the gradients created in the previous step
            optimizer.step()

            
            print("Epoch {}, Batch {}, Content Loss {}, Style Loss {}, Total Loss {}".format(epoch, batch, c_loss, s_loss, total_loss))

            if batch % 100 == 0:
                
                print_img = torch.cat((content[:1], style[:1], final_image[:1]), 3).detach().cpu()
                concat_img(print_img)
        


if __name__ == "__main__":
    train()