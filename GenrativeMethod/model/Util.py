import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    SpatialPadd, RandSpatialCropd, RandAffined, ToTensord,
    RandRotated, RandZoomd, RandFlipd, CropForegroundd, RandCropByPosNegLabeld,Resized
)
from monai.transforms import Transform
from monai.transforms.utils import equalize_hist

import torch
import numpy as np
from scipy.ndimage import label
from numba import jit

from AttenUnet.AttenUnet_4M import Att_Unet

class CustomEqualizeHist(Transform):
    def __init__(self, keys, num_bins='auto', min=0, max=1):
        self.keys = keys
        self.num_bins = num_bins
        self.min = min
        self.max = max

    def __call__(self, data):
        d = dict(data)

        # label = d['label']
        # label_ = np.array(label)
        # import matplotlib.pyplot as plt
        # for i in range(180):
        #     if np.sum(label_[0,:,:,i]) > 100:
        #         print('i')
        #         print(i)
        #         print(np.sum(label_[0,:,:,i]))

        # for j in range(180):
        #     if np.sum(label_[0,:,j,:]) > 100:
        #         print(j)
        #         print(j)
        #         print(np.sum(label_[0,:,j,:]))
            
        # for k in range(180):
        #     if np.sum(label_[0,k,:,:]) > 100:
        #         print('k')
        #         print(k)
        #         print(np.sum(label_[0,k,:,:]) )

        # plt.imshow(label_[0,:,:,115], cmap='gray')  # Use grayscale color map for mask visualization
        # plt.axis('off')  # Turn off axis numbering and labels
        # plt.savefig(f'mask_slice_3.png', bbox_inches='tight', pad_inches=0)  # Save the plot as a PNG image
        # plt.close()

        for key in self.keys:
            img = d[key]
            mask = img > 0
            
            img_ = np.array(img)
            mask_ = np.array(mask)
            # print(mask.shape)
            # print(img.mean())
            # import matplotlib.pyplot as plt
            # plt.imshow(img_[0,:,:,115], cmap='gray')  # Use grayscale color map for mask visualization
            # plt.axis('off')  # Turn off axis numbering and labels
            # plt.savefig(f'mask_slice_1.png', bbox_inches='tight', pad_inches=0)  # Save the plot as a PNG image
            # plt.close()

            equalized_img_np = equalize_hist(img_, mask=mask_, num_bins=self.num_bins, min=self.min, max=self.max)

            # plt.imshow(equalized_img_np[0,:,:,115], cmap='gray')  # Use grayscale color map for mask visualization
            # plt.axis('off')  # Turn off axis numbering and labels
            # plt.savefig(f'mask_slice_2.png', bbox_inches='tight', pad_inches=0)  # Save the plot as a PNG image
            # plt.close()
            # assert False

             # Convert back to original data type and device
            equalized_img = torch.as_tensor(equalized_img_np, device=img.device if isinstance(img, torch.Tensor) else 'cpu')           
            d[key].tensor = equalized_img
        return d

def train_test_transform(args):

    # create test set transformation
    test_transform_list = [
        LoadImaged(keys=['ct', 'label']),
        AddChanneld(keys=['ct', 'label']),
        CropForegroundd(keys=['ct', 'label'], source_key='ct'),
        SpatialPadd(keys=['ct', 'label'], spatial_size=[160, 192, 176]) # original_shape = [189, 233, 197]
        # ScaleIntensityd(keys=['ct']),
    ]
    # create train set transformation
    train_transform_list = [
        LoadImaged(keys=['ct', 'label']),
        AddChanneld(keys=['ct', 'label']),
        CropForegroundd(keys=['ct', 'label'], source_key='ct'),
        SpatialPadd(keys=['ct', 'label'], spatial_size=[160, 192, 176]), # Can be deleted if using any resize
    ]

    if args.scale_intensity:
        train_transform_list.append(ScaleIntensityd(keys=['ct']))
    if args.histogram_equal:
        train_transform_list.append(CustomEqualizeHist(keys=['ct']))
        test_transform_list.append(CustomEqualizeHist(keys=['ct']))
    if args.RandCropByPosNegLabeld:
        train_transform_list.append(RandCropByPosNegLabeld(keys=['ct', 'label'], spatial_size=(100,100,100), label_key='label', pos=3,neg=1, num_samples=4))
    if args.rand_spatial_crop:
        train_transform_list.append(RandSpatialCropd(keys=["ct", "label"], roi_size=(20,20,20), random_size=True))
    if args.rand_affine:
        train_transform_list.append(RandAffined(keys=['ct', 'label'], 
                                                # rotate_range=(360, 360, 360),
                                                scale_range=(1, 5), 
                                                # translate_range=(5, 5, 5),
                                                mode=('bilinear', 'nearest'), 
                                                prob=0.5,
                                                padding_mode='border'))
    if args.resize:
        train_transform_list.append(Resized(keys=["ct", "label"], spatial_size=(96,96,96), mode=['area', 'nearest']))
    if args.flip:
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=0))
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=1))
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=2))
    if args.to_tensor: # default is true
        train_transform_list.append(ToTensord(keys=['ct', 'label']))
        test_transform_list.append(ToTensord(keys=['ct', 'label']))

    train_transforms = Compose(train_transform_list)
    test_transforms = Compose(test_transform_list)
    print(train_transform_list)
    print(test_transform_list)
    return train_transforms, test_transforms


def model_selection(args):
    if args.model_name == '3D_Unet':
        print('we are using 3D Unet model')
        model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2, 2),      
        )
        return model
    elif args.model_name == 'Att_Unet':
        print('We are using Attention Unet model')
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
        return model

    elif args.model_name == 'Att_Unet_Joe':
        print('We are using Attention Unet model from Rongxi and Joe Lee')
        model = Att_Unet()
        return model
    
    elif args.model_name == 'Swin_UnetR':
        print('we are using Swin_UnetR model')
        model = monai.networks.nets.SwinUNETR(img_size=(128, 128,128), in_channels=1, out_channels=1, feature_size=24)
        return model
    
    elif args.model_name == 'Unet++':
        print('we are using Unet++ model')
        model = monai.networks.nets.BasicUNetPlusPlus(spatial_dims = 3, 
                              out_channels = 1,
                              features=(32, 32, 64, 128, 256, 32),
                              deep_supervision = True
                              )
        return model
    else:
        assert False


def remove_small_lesions_5d_tensor(binary_label_5d_tensor, min_size=25, max_size= 10000000):
    """
    Remove small lesions from a 5D binary label tensor.

    Parameters:
    - binary_label_5d_tensor: 5D PyTorch tensor with shape (1, 1, D, H, W), the binary label of lesions.
    - min_size: int, minimum size of lesion to keep.

    Returns:
    - filtered_label_5d_tensor: 5D PyTorch tensor, the binary label with small lesions removed.
    """
    # Ensure tensor is on CPU and convert to NumPy array
    binary_label_3d_np = binary_label_5d_tensor.cpu().numpy()[0, 0]
    
    # Label connected components
    labeled_array, num_features = label(binary_label_3d_np)
    # print(f'Number of lesion is {num_features}')

    # Filter components using Numba-accelerated function
    keep_mask = filter_components(labeled_array, num_features, min_size, max_size)
    
    _, num_lesion_after_filter = label(keep_mask)
    # print(f'Number of lesion after filter is {num_lesion_after_filter}')

    # Convert the mask back to 5D NumPy array and then to PyTorch tensor
    filtered_label_5d_np = keep_mask.astype(np.uint8)[None, None, ...]  # Add back the batch and channel dimensions
    filtered_label_5d_tensor = torch.from_numpy(filtered_label_5d_np)

    return filtered_label_5d_tensor, num_lesion_after_filter

@jit(nopython=True)
def filter_components(labeled_array, num_features, min_size, max_size = 100000000):
    """
    Numba-accelerated function to filter out small components.

    Parameters:
    - labeled_array: 3D numpy array, labeled connected components.
    - num_features: int, number of connected components.
    - min_size: int, minimum size of components to keep.

    Returns:
    - A boolean array where True represents pixels to keep.
    """
    output = np.zeros(labeled_array.shape, dtype=np.bool_)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if component.sum() >= min_size and component.sum()<= max_size:
            output |= component
    return output