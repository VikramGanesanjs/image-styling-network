from net import StyleTransfer, content_loss, style_loss
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

mps_device = torch.device("mps")

# Define training constants, learning rate, and optimizer
LR = 1e-4
epochs = 1
style_transfer_network = StyleTransfer()
style_transfer_network.to(mps_device)
optimizer = torch.optim.Adam(style_transfer_network.decoder.parameters(), lr=LR)
BATCH_SIZE = 3


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
            print(content.device, style.device)

            # Reset the gradients so that they do not get too high
            decoder.zero_grad()
            print(content.type(), style.type())

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

            if batch % 10 == 0:
                print("Epoch {}, Batch {}, Content Loss {}, Style Loss {}, Total Loss {}".format(epoch, batch, c_loss, s_loss, total_loss))

            if batch % 100 == 0 and batch != 0:
                final_image_display = final_image.permute(1, 2, 0)
                content_image = content.permute(1, 2, 0)
                style_image = style.permute(1, 2, 0)
                final_image_display = final_image_display.cpu()

                content_image = content_image.cpu()
                style_image = style_image.cpu()

                plt.imshow(content_image)
                plt.show()

                plt.imshow(style_image)
                plt.show()

                plt.imshow(final_image_display)
                plt.show()



if __name__ == "__main__":
    train()