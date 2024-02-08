import torch
import wandb
import argparse
import random
from torch.utils.data import Subset
import numpy as np
import datetime
import os
import nibabel as nib
import sys
import json
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import monai

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference, SlidingWindowInferer

from torch.utils.data import DataLoader


from Util import train_test_transform, model_selection, remove_small_lesions_5d_tensor

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
    parser.add_argument('--flip', action='store_true',help='Flip along x,y,z')
    parser.add_argument('--rand_spatial_crop', action='store_true',help='Apply RandSpatialCropd transform')
    parser.add_argument('--crop_size', nargs=3, type=int, default=[96, 96, 96], help='Spatial size for RandSpatialCropd')
    parser.add_argument('--rand_affine', action='store_true', help='Apply RandAffined transform')
    parser.add_argument('--to_tensor', action='store_false', help='Apply ToTensord transform')


    # Training Parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    # Add more arguments as needed
    parser.add_argument('--resuming', action='store_true', help='Continue trianing from previous checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path for model checkpoint')

    return parser.parse_args()

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available.")
        assert False

    args = parse_args()

    # Training Parameters
    batch_size= 1

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
    train_data_dicts = [{'ct': ct, 'label': label, 'id': os.path.basename(ct)} for ct, label in train_pairs]
    test_data_dicts = [{'ct': ct, 'label': label, 'id': os.path.basename(ct)} for ct, label in test_pairs]
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    test_dataset = Dataset(data=test_data_dicts, transform=test_transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("Finish Constructing Dataset")

    ################ Model initialization ##################
    model = model_selection(args)
    model = torch.nn.DataParallel(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")


    ############### Loss and Optimizer   ###################
    test_loss = monai.losses.DiceLoss()

    ################ Resuming ####################
    assert args.checkpoint_path is not None
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda:0')

    model.load_state_dict(checkpoint['model_state_dict'])
    # best_test_loss = checkpoint['loss']
    model = model.to(device)
    print("Load saved model")

    # Testing loop
    model.eval()
    # model.deep_supervision = False
    filename_dice = f'./test_result/{args.model_name}_diceloss.json'
    file_name_data = f'./test_result/{args.model_name}_raw_data.json'
    dice_file = []
    data_file = []

    total_test_loss = 0
    test_samples = 0
    num_lesion = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            print(batch_idx)

            if args.rand_spatial_crop == True:
                pred = sliding_window_inference(sample['ct'].to(device), args.crop_size, 60, model)
            else:
                pred = model(sample['ct'].to(device))

            # pred = torch.sigmoid(pred[-1])      ###Unet++
            pred = torch.sigmoid(pred)
            pred = pred > (pred.max() + pred.min())/2

            label = sample['label'].to(device)

            # Filter out small lesion
            label, num_lesion_after_filter = remove_small_lesions_5d_tensor(label, min_size=8000)
            print(f'Number of lesion after filter is {num_lesion_after_filter}')
            if num_lesion_after_filter<1:
                continue

            pred, _ = remove_small_lesions_5d_tensor(pred, min_size=8000)
            label = label.to(device)
            pred = pred.to(device)

            num_lesion += num_lesion_after_filter
            loss_ = test_loss(pred, label)
            total_test_loss += loss_.item() * sample['ct'].size(0)
            test_samples += sample['ct'].size(0)

            dice_data = {
                "id": batch_idx,
                "name": sample['id'][0],
                "dice_loss": loss_.item(),
            }

            # raw_data = {
            #     "id": batch_idx,
            #     "name": sample['id'][0],
            #     "pred": pred.int().cpu().numpy().tolist(),
            #     "label": label.int().cpu().numpy().tolist(),
            # }

            affine_matrix = np.eye(4)
            pred_numpy = pred.int().cpu().numpy()
            nifti_img = nib.Nifti1Image(pred_numpy[0,0], affine=affine_matrix)
            nib.save(nifti_img, './test_result/predicted_labels.nii.gz')
            print(dice_data['name'])
            assert False

            dice_file.append(dice_data)
            # data_file.append(raw_data) # For some unknow reason, it is too large

        with open(filename_dice, 'w') as file:
            json.dump(dice_file, file, indent=4)
        print("Save dice loss")

        with open(file_name_data, 'w') as file:
            json.dump(data_file, file, indent=4)
        print("Save data file")

        avg_test_loss = total_test_loss / test_samples
        print(f'The final loss is {avg_test_loss}')
        print(f'Total number of lesion over the threshold is {num_lesion}')


if __name__ == '__main__':
    main()