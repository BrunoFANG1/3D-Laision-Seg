import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from CTDataset import CTDataset
from model.Unet import UNet3D
from pytorch3dunet.unet3d.losses import DiceLoss
from tqdm import tqdm
import os

def main(rank, world_size):
    setup(rank, world_size)

    # construct dataset
    train_set = CTDataset(CT_image_root="/home/bruno/xfang/dataset/images/", MRI_label_root="/home/bruno/xfang/dataset/labels/")
    train_sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(dataset=train_set, batch_size=1, sampler=train_sampler)

    model = UNet3D(in_channels=1, out_channels=1, num_levels=3)
    model = DDP(model.to(rank), device_ids=[rank])

    loss = DiceLoss(normalization='sigmoid')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 10
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # ensure shuffling is per-epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        cumulative_loss = 0.0
        ct_images_processed = 0

        for ctid, mriid, img, mask in progress_bar:
            optimizer.zero_grad()
            img, mask = img.to(rank), mask.to(rank)
            output = model(img)
            loss_ = loss(output, mask)
            loss_.backward()
            optimizer.step()

            cumulative_loss += loss_.item() * img.size(0)
            ct_images_processed += img.size(0)
            average_loss_per_ct_image = cumulative_loss / ct_images_processed
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}, Average Loss per CT Image: {average_loss_per_ct_image:.4f}")

    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size+1} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
