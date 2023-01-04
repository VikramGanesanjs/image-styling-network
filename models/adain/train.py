from net import StyleTransfer, content_loss, style_loss
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define training constants, learning rate, and optimizer
LR = 1e-4
epochs = 1000
style_transfer_network = StyleTransfer()
optimizer = torch.nn.optim.Adam(style_transfer_network.decoder.parameters, lr=LR)
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

for epoch in range(epochs):
    #train the decoder initially
    decoder.train()

    for batch, (content) in enumerate(content_data_loader):
        style, _ = next(iter(style_data_loader))

        # Reset the gradients so that they do not get too high
        decoder.zero_grad()

        # Run the content and style through the model
        style_content_combined, encoded_style, encoded_content, final_image = style_transfer_network.forward(content, style)


        # Compute losses as outlined in the paper
        c_loss = content_loss(encoded_content[3]. style_content_combined)
        s_loss = style_loss(encoded_content, encoded_style)

        # Total loss weights style_loss doubly
        total_loss = content_loss + 2 * style_loss

        # Backpropagate through the network
        total_loss.backward()
        # Change the weights according to the gradients created in the previous step
        optimizer.step()

        if batch % 10 == 0:
            print("Epoch {}, Batch {}, Content Loss {}, Style Loss {}, Total Loss {}")

        if batch % 100 == 0:
            final_image_display = final_image.permute(0, 2, 3, 1)
            content_image = content.permute(0, 2, 3, 1)
            style_image = style.permute(0, 2, 3, 1)

            plt.imshow(content_image)
            plt.show()

            plt.imshow(style_image)
            plt.show()

            plt.imshow(final_image)
            plt.show()
            