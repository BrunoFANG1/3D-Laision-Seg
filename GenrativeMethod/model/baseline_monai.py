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
from sklearn.model_selection import train_test_split
import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    SpatialPadd, RandSpatialCropd, RandAffined, ToTensord,
    RandRotated, RandZoomd
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference, SlidingWindowInferer

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss

import os
import sys

sys.path.append('./Diff_UNet/')
sys.path.append('./Diff_UNet/BraTS2020')
from BraTS2020.train import DiffUNet



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

    # create train set transformation
    train_transform_list = [
        LoadImaged(keys=['ct', 'label']),
        AddChanneld(keys=['ct', 'label'])
    ]
    if args.scale_intensity:
        train_transform_list.append(ScaleIntensityd(keys=['ct']))
    if args.spatial_pad:
        original_shape = [189, 233, 197]
        assert sum([l>=r for l, r in zip(args.padding_size, original_shape)]) == 3, "Padding size must be greater than the original shape in all dimensions."
        train_transform_list.append(SpatialPadd(keys=['ct', 'label'], spatial_size=[224, 224, 224], mode='edge'))
    if args.rand_affine:
        train_transform_list.append(RandAffined(keys=['ct', 'label'], 
                                                # rotate_range=(360, 360, 360),
                                                # scale_range=(0.9, 1.1), 
                                                # translate_range=(5, 5, 5),
                                                mode=('bilinear', 'nearest'), 
                                                prob=1.0,
                                                padding_mode='border'))
    if args.rand_spatial_crop:
        train_transform_list.append(RandSpatialCropd(keys=["ct", "label"], roi_size=args.crop_size, random_size=False))
    if args.to_tensor: # default is true
        train_transform_list.append(ToTensord(keys=['ct', 'label']))
    train_transforms = Compose(train_transform_list)

    # create test set transformation
    test_transforms = Compose([
        LoadImaged(keys=['ct', 'label']),
        AddChanneld(keys=['ct', 'label']),
        # ScaleIntensityd(keys=['ct']),
        ToTensord(keys=['ct', 'label'])
    ])

    # Split train and test set
    # Directory paths
    ct_dir = "/home/bruno/xfang/dataset/images/"
    label_dir = "/home/bruno/xfang/dataset/labels/"

    # Pair each CT image with its corresponding label
    ct_img_label_pairs = [
        (os.path.join(ct_dir, f), 
        os.path.join(label_dir, f.replace('ct.nii.gz', 'label_inskull.nii.gz')))
        for f in os.listdir(ct_dir) if f.endswith('.nii.gz')
    ]

    # Split the pairs into training and testing sets
    train_pairs, test_pairs = train_test_split(ct_img_label_pairs, test_size=0.2)

    # Create data dictionaries
    train_data_dicts = [{'ct': ct, 'label': label} for ct, label in train_pairs]
    test_data_dicts = [{'ct': ct, 'label': label} for ct, label in test_pairs]

    # Build dataset for distributed GPUs
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    test_dataset = Dataset(data=test_data_dicts, transform=test_transforms)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    # Model initialization
    if args.model_name == '3D_Unet':
        model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True).to(rank)
    elif args.model_name == 'Att_Unet':
        model = monai.networks.nets.AttentionUnet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 1,
        channels = [16, 32, 64, 128],
        strides = [2,2,2,2],
        kernel_size = 3,
        up_kernel_size = 3,
        dropout = 0.0,
        )
    elif args.model_name == 'Swin_UnetR':
        model = monai.networks.nets.SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=1, feature_size=48)
    
    elif args.model_name == 'Diff_Unet':
        model = DiffUNet()

    else:
        assert False

    scaler = torch.cuda.amp.GradScaler()
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")
    model = DDP(model.to(rank), device_ids=[rank])


    # loss and optimizer selection
    train_loss = monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)
    test_loss = monai.losses.DiceLoss()
    # adversarial_loss =torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # log all info and parameters
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

        # log info
        print(f"The current epoch is {epoch}")
        if rank == 0 and args.wandb:
            wandb.log({"epoch": epoch})

        # Training loop
        model.train()
        total_train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()
        avg_test_loss = 0.0

        for batch_idx, sample in enumerate(train_loader):

            # regular training
            label = sample['label'].to(rank)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(sample['ct'].to(rank))
                loss_ = train_loss(pred, label) / gradient_accumulation_steps
            
            scaler.scale(loss_).backward()
            total_train_loss += loss_.item() * sample['ct'].size(0) * gradient_accumulation_steps
            train_samples += sample['ct'].size(0)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()


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

                if args.rand_spatial_crop == True:
                    pred = sliding_window_inference(sample['ct'].to(rank), args.crop_size, 20, model)
                else:
                    pred = model(sample['ct'].to(rank))

                # add Rongxi and Joe's evaluation
                pred = torch.sigmoid(pred)
                pred = pred > (pred.max() + pred.min())/2

                label = sample['label'].to(rank)
                loss_ = test_loss(pred, label)
                total_test_loss += loss_.item() * sample['ct'].size(0)
                test_samples += sample['ct'].size(0)

        # Save the model checkpoint with lowest test loss
        if rank == 0 and avg_test_loss <= total_test_loss / test_samples:
            PATH = f'/home/bruno/3D-Laision-Seg/GenrativeMethod/model/model_checkpoint/{args.model_name}_{epochs}_epoch.pth'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, PATH)

        avg_test_loss = total_test_loss / test_samples
        if rank == 0 and args.wandb:
            print(f"Average Test Loss for Epoch {epoch}: {avg_test_loss}")
            wandb.log({"epoch": epoch, "average_test_loss": avg_test_loss})

        

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
    
    parser.add_argument('--scale_intensity', action='store_true', help='Apply ScaleIntensityd transform')
    parser.add_argument('--spatial_pad', action='store_true', help='Apply SpatialPadd transform')
    parser.add_argument('--padding_size', nargs=3, type=int, default=[224, 224, 224], help='Padding size')
    parser.add_argument('--rand_spatial_crop', action='store_true',help='Apply RandSpatialCropd transform')
    parser.add_argument('--crop_size', nargs=3, type=int, default=[96, 96, 96], help='Spatial size for RandSpatialCropd')
    parser.add_argument('--rand_affine', action='store_true', help='Apply RandAffined transform')
    parser.add_argument('--to_tensor', action='store_false', help='Apply ToTensord transform')


    # Training Parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    # Add more arguments as needed

    return parser.parse_args()