import os
import argparse
import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

# Argument parsing
parser = argparse.ArgumentParser(description='3D Brain CT Image Preprocessing')
parser.add_argument('--add_channel', action='store_true', help='Add a channel dimension')
parser.add_argument('--scale_intensity', action='store_true', help='Scale intensity')
parser.add_argument('--resize', action='store_true', help='Resize images')
parser.add_argument('--to_tensor', action='store_true', help='Convert to tensor')
args = parser.parse_args()

# Directory paths
image_dir = '/home/bruno/xfang/dataset/images'
label_dir = '/home/bruno/xfang/dataset/labels'

# Automatically pair images with labels
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
labels = [os.path.join(label_dir, f.replace('_ct.nii.gz', '_label_inskull.nii.gz')) for f in images]

# Define your transforms based on argparse
transforms_list = [LoadImaged(keys=['image', 'label'])]
if args.add_channel:
    transforms_list.append(AddChanneld(keys=['image', 'label']))
if args.scale_intensity:
    transforms_list.append(ScaleIntensityd(keys=['image']))
if args.resize:
    transforms_list.append(Resized(keys=['image', 'label'], spatial_size=(96, 96, 96)))
if args.to_tensor:
    transforms_list.append(ToTensord(keys=['image', 'label']))

transforms = Compose(transforms_list)

# Create a dataset and dataloader
data_dicts = [{'image': image, 'label': label} for image, label in zip(images, labels)]
dataset = Dataset(data=data_dicts, transform=transforms)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterating through the DataLoader
for batch in dataloader:
    images, labels = batch['image'], batch['label']
    # Your model training code goes here
