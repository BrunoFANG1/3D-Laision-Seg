import torch
import wandb
import argparse
import random
from torch.utils.data import Subset
import numpy as np
import itertools
import sys
from AttenUnet.AttenUnet_4M import Att_Unet
sys.path.append('../')
from CTDataset import StrokeAI


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss
import os
import json


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
    args = parse_args()
    print(args)
    args_dict = vars(args)

    # Training Parameters
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size= args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    # Dataset Parameters
    # construct train dataset and test dataset
    dataset = StrokeAI(CT_root="/home/bruno/xfang/dataset/images",
                       DWI_root="/scratch4/rsteven1/DWI_coregis_20231208",  #DWI
                       ADC_root="/scratch4/rsteven1/ADC_coregis_20231228",  # ADC
                       label_root="/home/bruno/xfang/dataset/labels", 
                       MRI_type = 'ADC',
                       map_file= "/home/bruno/3D-Laision-Seg/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json",
                       bounding_box=args.bounding_box,
                       instance_normalize=args.instance_normalize, 
                       padding=args.padding, 
                       slicing=args.slicing,
                       crop=args.crop,
                       RotatingResize = args.RotatingResize)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42, indices_file='../dataset_indices.json')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    # Model initialization
    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True).to(rank)
    # model = Att_Unet()
    model = DDP(model.to(rank), device_ids=[rank])
    # discriminator = DISCRIMINATOR().to(rank)

    # discriminator = DDP(discriminator.to(rank), device_ids=[rank])

    # Add argparse
    # saved_model_path = '/home/bruno/3D-Laision-Seg/GenrativeMethod/model/model_checkpoint/Atten_4M__Joe_CT_100_epoch.pth'
    # checkpoint = torch.load(saved_model_path, map_location=lambda storage, loc: storage.cuda(rank))
    # model.load_state_dict(checkpoint)
    # print("load saved model")


    # loss and optimizer selection
    loss = DiceLoss(normalization='none')
    # adversarial_loss =torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if rank == 0 and args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="StrokeAI",

            name=args.model_name,
            
            # track hyperparameters and run metadata
            config = args_dict
        )

    print("start training")
    for epoch in range(epochs):
        print(f"The current epoch is {epoch}")
        if rank == 0 and args.wandb:
            wandb.log({"epoch": epoch})


        # Training loop
        model.train()
        total_train_loss = 0
        train_samples = 0
        optimizer.zero_grad()

        for batch_idx, sample in enumerate(train_loader):
            
            pred = model(sample['ct'].to(rank))
            label = sample['label'].to(rank)
            loss_ = loss(pred, label) / gradient_accumulation_steps
            loss_.backward()
            total_train_loss += loss_.item() * sample['ct'].size(0) * gradient_accumulation_steps
            train_samples += sample['ct'].size(0)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = reduce_loss(total_train_loss / train_samples, rank, world_size)
        if rank == 0 and args.wandb:
            print(f"Average Training Loss for Epoch {epoch}: {avg_train_loss}")
            wandb.log({"epoch": epoch, "average_train_loss": avg_train_loss})

        # Testing loop
        model.eval()
        total_test_loss = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                pred = model(sample['ct'].to(rank))

                # add Rongxi and Joe's evaluation
                pred = pred > (pred.max() + pred.min())/2

                label = sample['label'].to(rank)
                loss_ = loss(pred, label)
                total_test_loss += loss_.item() * sample['ct'].size(0)
                test_samples += sample['ct'].size(0)

        avg_test_loss = total_test_loss / test_samples
        if rank == 0 and args.wandb:
            print(f"Average Test Loss for Epoch {epoch}: {avg_test_loss}")
            wandb.log({"epoch": epoch, "average_test_loss": avg_test_loss})

        
    print("training complete and save the model")
    PATH = f'/home/bruno/3D-Laision-Seg/GenrativeMethod/model/model_checkpoint/{args.model_name}_{epochs}_epoch.pth'
    
    if rank == 0:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, PATH)

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


def parse_args():
    parser = argparse.ArgumentParser(description="StrokeAI Training Script")

    # Model Parameters
    ### You have to give a name of the model name
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')

    # log info
    parser.add_argument('--wandb', action='store_true', help='log data to wandb')

    # Dataset Parameters
    parser.add_argument('--CT_root', type=str, default='/home/bruno/xfang/dataset/images', help='Root directory for CT images')
    parser.add_argument('--DWI_root', type=str, default='/scratch4/rsteven1/DWI_coregis_20231208', help='Root directory for DWI images')
    parser.add_argument('--ADC_root', type=str, default='/scratch4/rsteven1/ADC_coregis_20231228', help='Root directory for ADC images')
    parser.add_argument('--label_root', type=str, default='/home/bruno/xfang/dataset/labels', help='Root directory for label images')
    parser.add_argument('--MRI_type', type=str, default='ADC', choices=['ADC', 'DWI', 'Other'], help='Type of MRI images')
    parser.add_argument('--map_file', type=str, default= "/home/bruno/3D-Laision-Seg/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json", help='Path to the map file')
    parser.add_argument('--bounding_box', action='store_true', help='Whether to use bounding box')
    parser.add_argument('--padding', action='store_true', help='Whether to use padding')
    parser.add_argument('--slicing', action='store_true', help='Whether to use slicing')
    parser.add_argument('--instance_normalize', action='store_true', help='Whether to use instance normalization')
    parser.add_argument('--crop', action='store_true', help='Crop size, if any(need to upgrade)')
    parser.add_argument('--RotatingResize', action='store_true', help='Whether to use rotating resize')


    # Training Parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of gradient accumulation steps')

    # Add more arguments as needed



    return parser.parse_args()