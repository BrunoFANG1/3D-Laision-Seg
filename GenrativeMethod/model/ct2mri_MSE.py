import torch
import wandb
import random
from torch.utils.data import Subset
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D,ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss
import os

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
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    # construct train dataset and test dataset
    dataset = StrokeAI(CT_root="../dataset/images/", MRI_root="/scratch4/rsteven1/New_MRI/", label_root="../dataset/labels/", bounding_box=True, normalize=True, padding=False)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

    # Model initialization
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True)
    model = DDP(model.to(rank), device_ids=[rank])

    # loss and optimizer selection
    learning_rate = 1e-3
    # loss = DiceLoss(normalization='none')
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    grad_accumulation_steps = 4
    optimizer.zero_grad() 

    epochs = 10

    if rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="StrokeAI",

            name="test_run",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": "3D_Unet",
            "dataset": "CT+MRI+label",
            "epochs": epochs,
            }
        )


    print("start training")
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        train_sampler.set_epoch(epoch)


        if rank == 0:
            wandb.log({"epoch": epoch})

        for batch_idx, sample in enumerate(train_loader):

            ct, mri, _ = sample['ct'].to(rank), sample['mri'].to(rank), sample['label'].to(rank)

            generated_mri = model(ct)
            loss_ = loss(generated_mri, mri)
            loss_ = loss_ / grad_accumulation_steps
            loss_.backward()

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # calculate average loss and log into wandb, note I'm using distributed training
            total_loss += loss_.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # LOG loss per epoch
        reduced_avg_loss = reduce_loss(avg_loss, rank, world_size)
        if rank == 0:  # Log only from the first process
            wandb.log({"epoch": epoch, "average_loss": reduced_avg_loss})

    print("training complete and save the model")
    PATH = './model/trained_Unet3D_4_layer.pth'
    if rank == 0:
        torch.save(model.state_dict(), PATH)
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
