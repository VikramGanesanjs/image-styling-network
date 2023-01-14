from net import StyleTransfer
import torch
import torch.nn as nn
from pathlib import Path
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.multiprocessing
from utils import *
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from decoder import decoder as Decoder
from encoder import encoder as Encoder
from PIL import Image, ImageFile

class FlatFolderDataset(data.Dataset):
        def __init__(self, root, transform):
            super(FlatFolderDataset, self).__init__()
            self.root = root
            self.paths = list(Path(self.root).glob('*'))
            self.transform = transform

        def __getitem__(self, index):
            path = self.paths[index]
            img = Image.open(str(path)).convert('RGB')
            img = self.transform(img)
            return img

        def __len__(self):
            return len(self.paths)

        def name(self):
            return 'FlatFolderDataset'

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set the path to the dataset directory
    content_dataset_dir = '../../content-dataset/images/images'
    style_dataset_dir = '../../style-dataset/images'


    def train_transform():
        transform_list = [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)


    

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', default=content_dataset_dir, type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', default=style_dataset_dir, type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--encoder', type=str, default='./vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', default='../saved-models',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--save_model_interval', type=int, default=500)
    parser.add_argument('--save-image-interval', type=int, default=50)
    args = parser.parse_args()




    device = torch.device('mps')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))


    decoder = Decoder
    encoder = Encoder

    encoder.load_state_dict(torch.load(args.encoder))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    network = StyleTransfer(encoder, decoder)
    network.train()
    network.to(device)

    content_dataset = FlatFolderDataset(args.content_dir, transform=train_transform())
    style_dataset = FlatFolderDataset(args.style_dir, transform=train_transform())

    print(len(content_dataset), len(style_dataset))

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        num_workers=args.n_threads))
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)


    for batch in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, batch, args.lr_decay, args.lr)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        final_image, s_loss, c_loss = network(content_images, style_images)
        c_loss = args.content_weight * c_loss
        s_loss = args.style_weight * s_loss
        total_loss = c_loss + s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', c_loss.item(), batch + 1)
        writer.add_scalar('loss_style', s_loss.item(), batch + 1)

        if (batch + 1) % args.save_model_interval == 0 or (batch + 1) == args.max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'decoder_iter_{:d}.pth.tar'.format(batch + 1))

        if (batch + 1) % args.save_image_interval == 0:
            print_img = torch.cat((content_images[:1], style_images[:1], final_image[:1]), 3).detach().cpu()
            concat_img(print_img, batch)
    writer.close()


if __name__ == "__main__":
    main()