import torch
import wandb
import argparse
import random
from torch.utils.data import Subset
import numpy as np
import datetime
import os
import sys
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import monai

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference, SlidingWindowInferer

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from Util import train_test_transform, model_selection



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_loss(loss, sample_num, rank, world_size):
    """
    Reduce the loss across all processes so that every process has the average loss.

    Args:
    loss (float): The loss to reduce per process.
    sample_num(int): The num of samples per process
    rank (int): The rank of the current process in the distributed training setup.
    world_size (int): The total number of processes in the distributed training setup.

    Returns:
    float: The reduced loss.
    """
    reduced_loss = torch.tensor(loss, device="cuda:{}".format(rank))
    reduced_sample_num =  torch.tensor(sample_num, device="cuda:{}".format(rank))
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_sample_num, op=dist.ReduceOp.SUM)
    loss_all_process = reduced_loss/reduced_sample_num
    return loss_all_process.item()


def main(rank, world_size):

    setup(rank, world_size)
    Default_config = OmegaConf.load('./config/config.yaml')
    input_config = OmegaConf.from_cli()
    args = OmegaConf.merge(Default_config, input_config)
    print(args)
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M")
    args_dict = OmegaConf.to_container(args, resolve=True)

    ################# Training Parameters ########################
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size= args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    ################# Dataset ###################
    train_transforms, test_transforms = train_test_transform(args)
    ct_dir = "/home/bruno/xfang/dataset/images/"
    label_dir = "/home/bruno/xfang/dataset/labels/"
    ct_img_label_pairs = [
        (os.path.join(ct_dir, f), 
        os.path.join(label_dir, f.replace('ct.nii.gz', 'label_inskull.nii.gz')))
        for f in os.listdir(ct_dir) if f.endswith('.nii.gz')
    ]
    train_pairs, test_pairs = train_test_split(ct_img_label_pairs, test_size=0.2, random_state=42)
    train_data_dicts = [{'ct': ct, 'label': label} for ct, label in train_pairs]
    test_data_dicts = [{'ct': ct, 'label': label} for ct, label in test_pairs]
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    test_dataset = Dataset(data=test_data_dicts, transform=test_transforms)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=8)

    ################ Model initialization ##################
    model = model_selection(args)
    scaler = torch.cuda.amp.GradScaler()    # Will that affect model accuracy?
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")
    model = DDP(model.to(rank), device_ids=[rank])


    ############### Loss and Optimizer   ###################
    # train_loss = monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)
    train_loss = monai.losses.DiceLoss(sigmoid=True)
    test_loss = monai.losses.DiceLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

    ################ Resuming ####################
    if args.resuming:
        checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # best_test_loss = checkpoint['loss']
        print("Load saved model and optimizer")


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
        # model.deep_supervision = True # only for Unet++
        total_train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()
        best_test_loss = 1.0

        for batch_idx, sample in enumerate(train_loader):

            # regular training
            label = sample['label'].to(rank)
            optimizer.zero_grad()

            # with torch.cuda.amp.autocast():                                   # Half precision
            #     pred = model(sample['ct'].to(rank))                           # Half precision
            #     loss_ = train_loss(pred, label) / gradient_accumulation_steps # Half precision
                # loss_ = (train_loss(pred[0], label) + train_loss(pred[1], label) + train_loss(pred[2], label) + train_loss(pred[3], label)) / gradient_accumulation_steps
            pred = model(sample['ct'].to(rank))
            loss_ = train_loss(pred, label) / gradient_accumulation_steps

            # scaler.scale(loss_).backward()                                     # Half precision
            loss_.backward()
            total_train_loss += loss_.item() * sample['ct'].size(0) * gradient_accumulation_steps
            train_samples += sample['ct'].size(0)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # scaler.step(optimizer)                           # Half precision
                # scaler.update()                           # Half precision
                optimizer.step()

        avg_train_loss = reduce_loss(total_train_loss, train_samples, rank, world_size)
        if rank == 0 and args.wandb:
            print(f"Average Training Loss for Epoch {epoch}: {avg_train_loss}")
            wandb.log({"epoch": epoch, "average_train_loss": avg_train_loss})

        # Testing loop
        model.eval()
        # model.deep_supervision = False
        total_test_loss = 0
        test_samples = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):

                if args.rand_spatial_crop == True:
                    pred = sliding_window_inference(sample['ct'].to(rank), args.crop_size, 60, model)
                else:
                    pred = model(sample['ct'].to(rank))

                # add Rongxi and Joe's evaluation
                # pred = torch.sigmoid(pred[-1])      ###Unet++
                pred = torch.sigmoid(pred)
                pred = pred > (pred.max() + pred.min())/2

                label = sample['label'].to(rank)
                loss_ = test_loss(pred, label)
                total_test_loss += loss_.item() * sample['ct'].size(0)
                test_samples += sample['ct'].size(0)

        # Save the model checkpoint with lowest test loss
        avg_test_loss = reduce_loss(total_test_loss, test_samples, rank, world_size)
        if rank == 0 and best_test_loss <= avg_test_loss:
            PATH = f'/home/bruno/3D-Laision-Seg/GenrativeMethod/model/model_checkpoint/{args.model_name}_{epochs}_epoch_{timestamp}.pth'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, PATH)

        if rank == 0 and args.wandb:
            print(f"Average Test Loss for Epoch {epoch}: {avg_test_loss}")
            wandb.log({"epoch": epoch, "average_test_loss": avg_test_loss})

        

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)