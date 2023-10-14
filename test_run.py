import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from CTDataset import CTDataset
from model.Unet import UNet3D
from torch.utils.tensorboard import SummaryWriter
from pytorch3dunet.unet3d.losses import DiceLoss
from tqdm import tqdm
import os
from model.SwinUnet.config import get_config

# models
from model.SwinUnet.networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)





def main(rank, world_size):
    setup(rank, world_size)
    writer = SummaryWriter()

    # construct dataset
    all_indices = list(range(len(os.listdir("/home/bruno/xfang/dataset/images/"))))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    train_set = CTDataset(CT_image_root="/home/bruno/xfang/dataset/images/", 
                          MRI_label_root="/home/bruno/xfang/dataset/labels/", 
                          indices=train_indices,
                          padding = True)
    test_set = CTDataset(CT_image_root="/home/bruno/xfang/dataset/images/", 
                         MRI_label_root="/home/bruno/xfang/dataset/labels/", 
                         indices=test_indices,
                         padding = True)

    train_sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(dataset=train_set, batch_size=4, sampler=train_sampler)
    test_sampler = DistributedSampler(test_set)
    test_dataloader = DataLoader(dataset=test_set, batch_size=2, sampler=test_sampler)


    # model = UNet3D(in_channels=1, out_channels=1, num_levels=4)
    model = ViT_seg(config, num_classes=1, img_size=190)

    model = DDP(model.to(rank), device_ids=[rank])

    loss = DiceLoss(normalization='sigmoid')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 20
    for epoch in range(epochs):

        model.train()
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
            writer.add_scalar('Average Training Loss', average_loss_per_ct_image, epoch)
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
