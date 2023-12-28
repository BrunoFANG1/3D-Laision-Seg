import torch
import wandb
import random
from torch.utils.data import Subset
import numpy as np
import itertools

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D,ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss
import os

class DISCRIMINATOR(torch.nn.Module):
    def __init__(self):
        super(DISCRIMINATOR, self).__init__()
        self.model = UNet3D(in_channels=1, out_channels=1, num_levels=2, is_segmentation=False)
        self.conv = torch.nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, padding=1) 
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.conv(x)
        x = self.pool(x) 
        x = self.flatten(x)
        return x


def train_test_split(dataset, test_size=0.2, random_seed=42):
    """
    Splits the dataset into a training set and a test set.

    Args:
    dataset (Dataset): The dataset to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_seed (int): Random seed for reproducibility.

    Returns:
    train_dataset, test_dataset: Two datasets, for training and testing.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Generate a shuffled list of indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))
    random.shuffle(indices)

    # Split indices
    train_indices, test_indices = indices[split:], indices[:split]

    # Create two subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

# sometimes, the world_size is not equal to the real size of data
def reduce_loss(loss, rank, world_size):
    """
    Reduce the loss across all processes so that every process has the average loss.

    Args:
    loss (float): The loss to reduce.
    rank (int): The rank of the current process in the distributed training setup.
    world_size (int): The total number of processes in the distributed training setup.

    Returns:
    float: The reduced loss.
    """
    reduced_loss = torch.tensor(loss).to(rank)
    dist.reduce(reduced_loss, dst=0)
    if rank == 0:
        reduced_loss = reduced_loss / world_size
    return reduced_loss.item()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_loss(loss, rank, world_size):
    """
    Reduce the loss across all processes so that every process has the average loss.

    Args:
    loss (float): The loss to reduce.
    rank (int): The rank of the current process in the distributed training setup.
    world_size (int): The total number of processes in the distributed training setup.

    Returns:
    float: The reduced loss.
    """
    reduced_loss = torch.tensor(loss, device="cuda:{}".format(rank))
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss = reduced_loss / world_size
    return reduced_loss.item()


def main(rank, world_size):
    setup(rank, world_size)

    batch_size=1
    learning_rate = 1e-3
    epochs = 10

    if rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="StrokeAI",

            name="GAN_1",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": "3D_Unet",
            "dataset": "CT+MRI+label",
            "epochs": epochs,
            }
        )

    # construct train dataset and test dataset
    dataset = StrokeAI(CT_root="../dataset/images/", MRI_root="/scratch4/rsteven1/New_MRI/", label_root="../dataset/labels/", bounding_box=True, normalize=True, padding=False)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

    # Model initialization
    generator = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True).to(rank)
    discriminator = DISCRIMINATOR().to(rank)
    generator = DDP(generator.to(rank), device_ids=[rank])
    discriminator = DDP(discriminator.to(rank), device_ids=[rank])

    # loss and optimizer selection
    # loss = DiceLoss(normalization='none')
    adversarial_loss =torch.nn.CrossEntropyLoss()


    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


#### consider mix a batch with real mri and generated mri
    print("start training")
    for epoch in range(epochs):
        if rank == 0:
            wandb.log({"epoch": epoch})

        gen_loss = 0.0
        dis_loss = 0.0

        for batch_idx, sample in enumerate(train_loader):

            print(batch_idx)
            ### Train Generator ###
            optimizer_G.zero_grad()

            generated_mri = generator(sample['ct'].to(rank))
            fake_labels = torch.zeros(batch_size, dtype=torch.long).to(rank)

            fake_pred = discriminator(generated_mri)

            g_loss = adversarial_loss(fake_pred, fake_labels.to(rank))

            g_loss.backward()
            optimizer_G.step()
            avg_g_loss = reduce_loss(g_loss.item(), rank, world_size)
            gen_loss += avg_g_loss

            ### Train Discriminator ###
            optimizer_D.zero_grad()

            real_mri = sample['mri'].to(rank)
            real_labels = torch.ones(batch_size, dtype=torch.long).to(rank)

            # loss for real mri
            real_pred = discriminator(real_mri)
            d_real_loss = adversarial_loss(real_pred, real_labels)

            # loss for fake mri >> Why cal fake_pred again?
            fake_pred = discriminator(generated_mri.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step() 

            avg_d_loss = reduce_loss(d_loss.item(),rank, world_size)
            dis_loss += avg_d_loss
        
        if rank == 0:
            wandb.log({"gen_loss": gen_loss})
        if rank == 0:
            wandb.log({"dis_loss": dis_loss})

    print("training complete and save the model")
    PATH = './model/GAN_1.pth'
    if rank == 0:
        torch.save(generator.state_dict(), PATH)
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
