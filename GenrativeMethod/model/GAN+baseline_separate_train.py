import torch
import wandb
import random
from torch.utils.data import Subset
import numpy as np
import itertools
import wandb

import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory by going up one level
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D,ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss
import os
import json
from model.CT2MRI_3DGAN.model import StarGenerator3D

def save_indices(indices, file_path):
    with open(file_path, 'w') as file:
        json.dump(indices, file)

def load_indices(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def train_test_split(dataset, test_size=0.2, random_seed=42, indices_file=None):
    """
    Make sure the reproductibility and consistence of each experiment: Test set remains the same
    
    Args: 
    indices_file: indices for test dataset
    
    Returns:
    Same train dataset and test dataset for each run
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Check if indices file exists
    if indices_file and os.path.exists(indices_file):
        indices = load_indices(indices_file)
    else:
        # Generate a shuffled list of indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        save_indices(indices, indices_file if indices_file else 'dataset_indices.json')

    split = int(np.floor(test_size * len(indices)))
    train_indices, test_indices = indices[split:], indices[:split]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

# sometimes, the world_size is not equal to the real size of data


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


    gradient_accumulation_steps = 1
    batch_size=10
    learning_rate = 1e-3
    epochs = 100

    # Initialize wandb
    if rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="3D_Unet",

            name="GAN+baseline",
            
            # track hyperparameters and run metadata
            config={
            "epochs": epochs,
            }
            )

    # construct train dataset and test dataset
    dataset =StrokeAI(CT_root="/home/bruno/xfang/dataset/images",
                       MRI_root="/scratch4/rsteven1/DWI_coregis_20231208", 
                       label_root="/home/bruno/xfang/dataset/labels", 
                       bounding_box=True,
                       normalize=True, 
                       padding=True, 
                       slicing=True)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42, 
                                                   indices_file='/home/bruno/xfang/GenrativeMethod/dataset_indices.json')
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, sampler=test_sampler)

    # Segmentation model
    model = UNet3D(in_channels=1, out_channels=1, num_levels=3).to(rank)
    # discriminator = DISCRIMINATOR().to(rank)
    model = DDP(model.to(rank), device_ids=[rank])
    # discriminator = DDP(discriminator.to(rank), device_ids=[rank])

    # Generative mdoel
    G_model = StarGenerator3D().to(rank)
    checkpoint_path = "/home/bruno/xfang/GenrativeMethod/model/CT2MRI_3DGAN/CT2MRI_3DGAN/models/135000-G.ckpt"
    G_model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    for param in G_model.parameters():
        param.requires_grad = False


    # loss and optimizer selection
    loss = DiceLoss(normalization='none')
    # adversarial_loss =torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))


    print("start training")
    for epoch in range(epochs):
        print(f"The current epoch is {epoch}")

        # Training loop
        model.train()
        total_train_loss = 0
        train_samples = 0
        optimizer.zero_grad()

        for batch_idx, sample in enumerate(train_loader):
            gen_MRI = G_model(sample['ct'].to(rank))
            pred = model(gen_MRI).to(rank)
            print(pred.max())
            import pdb
            pdb.set_trace()
            label = sample['label'].to(rank)
            loss_ = loss(pred, label) / gradient_accumulation_steps
            loss_.backward()
            total_train_loss += loss_.item() * sample['ct'].size(0) * gradient_accumulation_steps
            train_samples += sample['ct'].size(0)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if rank == 0:
                wandb.log({'epoch': epoch, 'batch_idx': batch_idx, 'train_loss': loss_.item()})

        avg_train_loss = reduce_loss(total_train_loss / train_samples, rank, world_size)
        if rank == 0:
            print(f"Average Training Loss for Epoch {epoch}: {avg_train_loss}")
            wandb.log({'epoch': epoch, 'avg_train_loss': avg_train_loss})

        # Testing loop
        model.eval()
        total_test_loss = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                gen_MRI = G_model(sample['ct'].to(rank))
                pred = model(gen_MRI).to(rank)
                label = sample['label'].to(rank)
                loss_ = loss(pred, label)
                total_test_loss += loss_.item() * sample['ct'].size(0)
                test_samples += sample['ct'].size(0)

        avg_test_loss = total_test_loss / test_samples
        if rank == 0:
            print(f"Average Test Loss for Epoch {epoch}: {avg_test_loss}")
            wandb.log({'epoch': epoch, 'avg_test_loss': avg_test_loss})

        
    PATH = './model/model_checkpoint/GAN_baseline_separate_100epoch.pth'
    print(f"training complete and save the model at {PATH}")
    if rank == 0:
        torch.save(model.state_dict(), PATH)
    cleanup()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
