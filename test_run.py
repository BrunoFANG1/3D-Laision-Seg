import torch
import argparse
import wandb
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

    wandb.init(
        # set the wandb project where this run will be logged
        project="3D_Unet",

        name="SwinUnet",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "SwinUnet_test",
        "dataset": "3D",
        "epochs": 20,
        }
    )

    # construct dataset
    all_indices = list(range(len(os.listdir("/home/bruno/xfang/dataset/images/"))))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    train_set = CTDataset(CT_image_root="/home/bruno/xfang/dataset/images/", 
                          MRI_label_root="/home/bruno/xfang/dataset/labels/", 
                          indices=train_indices,
                          padding = True,
                          slicing = True,
                          )
    test_set = CTDataset(CT_image_root="/home/bruno/xfang/dataset/images/", 
                         MRI_label_root="/home/bruno/xfang/dataset/labels/", 
                         indices=test_indices,
                         padding= True,
                         slicing = True,
                         )

    train_sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(dataset=train_set, batch_size=100, sampler=train_sampler)
    test_sampler = DistributedSampler(test_set)
    test_dataloader = DataLoader(dataset=test_set, batch_size=100, sampler=test_sampler)


    # model = UNet3D(in_channels=1, out_channels=1, num_levels=3)
    model = ViT_seg(config, num_classes=1, img_size=190)

    model = DDP(model.to(rank), device_ids=[rank])

    loss = DiceLoss(normalization='sigmoid')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    epochs = 50
    print("start training")
    for epoch in range(epochs):

        model.train()
        train_sampler.set_epoch(epoch)  # ensure shuffling is per-epoch

        cumulative_loss = 0.0
        ct_images_processed = 0

        wandb.log({"epoch": epoch})

        for ctid, mriid, img, mask in train_dataloader:
            optimizer.zero_grad()
            img, mask = img.to(rank), mask.to(rank)
            output = model(img)
            loss_ = loss(output, mask)
            loss_.backward()
            optimizer.step()

            cumulative_loss += loss_.item() * img.size(0)
            ct_images_processed += img.size(0)
            loss_sum = loss_.clone()
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            average_loss_per_ct_image = loss_sum.item() / world_size

            wandb.log({"average_loss_per_ct_image": average_loss_per_ct_image})

        if epoch % 2 == 1:
            model.eval()  # Set the model to evaluation mode
            cumulative_loss = 0.0  # Reset cumulative loss
            ct_images_processed = 0  # Reset count of images processed

            with torch.no_grad():  # Disable gradient computation
                for ctid, mriid, img, mask in test_dataloader:
                    img, mask = img.to(rank), mask.to(rank)
                    output = model(img)
                    loss_ = loss(output, mask)  # Compute loss
                    cumulative_loss += loss_.item() * img.size(0)
                    ct_images_processed += img.size(0)

            test_loss_epoch = cumulative_loss / ct_images_processed  # Corrected divisor

            wandb.log({"test loss": test_loss_epoch})

    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
