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
import json

from model.CT2MRI_3DGAN.model import StarGenerator3D
from model.CT2MRI_3DGAN.model import StarDiscriminator3D


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

def gradient_penalty(y, x, rank):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(rank)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def main(rank, world_size):
    setup(rank, world_size)

    gradient_accumulation_steps = 4
    batch_size=2
    learning_rate = 1e-3
    epochs = 250

    # construct train dataset and test dataset
    dataset = StrokeAI(CT_root="../dataset/images/", MRI_root="/scratch4/rsteven1/New_MRI/", label_root="../dataset/labels/", bounding_box=True,  normalize=True, padding=True, slicing=True)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42, indices_file='./dataset_indices.json')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, sampler=test_sampler)

    # Model initialization
    S_model = UNet3D(in_channels=1, out_channels=1, f_maps=128, num_levels=3).to(rank)
    Generator = StarGenerator3D()
    Discriminator = StarDiscriminator3D()
    S_model = DDP(S_model.to(rank), device_ids=[rank])
    G_model = DDP(Generator.to(rank), device_ids=[rank])
    D_model = DDP(Discriminator.to(rank), device_ids=[rank])

    # loss and optimizer selection
    L1_loss = torch.nn.L1Loss()
    seg_loss = DiceLoss(normalization='none')
    # adversarial_loss =torch.nn.CrossEntropyLoss()

    S_optimizer = torch.optim.Adam(S_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    print("start training")
    for epoch in range(epochs):
        print(f"The current epoch is {epoch}")

        # Training loop
        S_model.train()
        G_model.train()
        D_model.train()
        S_optimizer.zero_grad()
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()

        total_train_loss = 0
        train_samples = 0

        seg_train_loss = 0
        train_samples = 0
        for batch_idx, sample in enumerate(train_loader):
            
            real_CT = sample['ct'].to(rank).float()
            real_MRI = sample['mri'].to(rank).float()
            real_label = sample['label'].to(rank).float()

            # Train Discriminator

            # Discriminate real MRI
            out_src = D_model(real_MRI)
            d_loss_real = - torch.mean(out_src)

            # Discriminate fake MRI
            fake_MRI = G_model(real_CT)
            out_src = D_model(fake_MRI.detach())
            d_loss_fake = torch.mean(out_src)

            # Computer loss for gradient penalty
            alpha = torch.rand(real_MRI.size(0), 1, 1, 1, 1).to(rank)
            x_hat = (alpha * real_MRI.data + (1 - alpha) * real_CT.data).requires_grad_(True)
            out_src = D_model(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, rank)

            # Backward pass
            d_loss = (d_loss_real + d_loss_fake + 10 * d_loss_gp)/gradient_accumulation_steps
            S_optimizer.zero_grad()
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            d_loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                D_optimizer.step()

            # Train Generator
            if (batch_idx + 1) % 5 == 0:
                fake_MRI = G_model(real_CT)
                out_src = D_model(fake_MRI)
                out_seg = S_model(fake_MRI)

                g_loss_fake = - torch.mean(out_src)
                g_loss_L1 = L1_loss(fake_MRI, real_MRI) * 100
                g_seg_loss = seg_loss(out_seg, real_label) * 100
                g_loss = (g_loss_fake + g_loss_L1 + g_seg_loss)/gradient_accumulation_steps
                S_optimizer.zero_grad()
                G_optimizer.zero_grad()
                D_optimizer.zero_grad() 
                g_loss.backward()
                if (batch_idx + 1)//5 % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    G_optimizer.step()
                    S_optimizer.step()

                train_samples += sample['ct'].size(0)
                seg_train_loss += g_seg_loss.item()/100 * sample['ct'].size(0)

        avg_train_seg_loss = reduce_loss(seg_train_loss / train_samples, rank, world_size)
        if rank == 0:
            print(f"Average Training Loss for Epoch {epoch}: {avg_train_seg_loss}")

        # Testing loop
        G_model.eval()
        S_model.eval()
        total_test_loss = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                pred = G_model(sample['ct'].to(rank))
                label = sample['label'].to(rank)
                loss_ = seg_loss(pred, label)
                total_test_loss += loss_.item() * sample['ct'].size(0)
                test_samples += sample['ct'].size(0)

        avg_test_loss = total_test_loss / test_samples
        if rank == 0:
            print(f"Average Test Loss for Epoch {epoch}: {avg_test_loss}")

        
    print("training complete and save the model")
    PATH = './model/GAN+BASELINE.pth'
    if rank == 0:
        torch.save(G_model.state_dict(), PATH)
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
