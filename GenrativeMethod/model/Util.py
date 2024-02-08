import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    SpatialPadd, RandSpatialCropd, RandAffined, ToTensord,
    RandRotated, RandZoomd, RandFlipd
)

import torch
import numpy as np
from scipy.ndimage import label
from numba import jit



def train_test_transform(args):

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
                                                rotate_range=(360, 360, 360),
                                                scale_range=(0.8, 1.2), 
                                                # translate_range=(5, 5, 5),
                                                mode=('bilinear', 'nearest'), 
                                                prob=0.5,
                                                padding_mode='border'))
    if args.flip:
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=0))
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=1))
        train_transform_list.append(RandFlipd(keys=["ct", "label"], prob=0.5, spatial_axis=2))
    if args.rand_spatial_crop:
        train_transform_list.append(RandSpatialCropd(keys=["ct", "label"], roi_size=args.crop_size, random_size=False))
    if args.to_tensor: # default is true
        train_transform_list.append(ToTensord(keys=['ct', 'label']))

    train_transforms = Compose(train_transform_list)

    test_transforms = Compose([
        LoadImaged(keys=['ct', 'label']),
        AddChanneld(keys=['ct', 'label']),
        # ScaleIntensityd(keys=['ct']),
        ToTensord(keys=['ct', 'label'])
    ])
    print(train_transform_list)
    return train_transforms, test_transforms


def model_selection(args):
    if args.model_name == '3D_Unet':
        print('we are using 3D Unet model')
        return None
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


def remove_small_lesions_5d_tensor(binary_label_5d_tensor, min_size=25):
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
    keep_mask = filter_components(labeled_array, num_features, min_size)
    
    _, num_lesion_after_filter = label(keep_mask)
    # print(f'Number of lesion after filter is {num_lesion_after_filter}')

    # Convert the mask back to 5D NumPy array and then to PyTorch tensor
    filtered_label_5d_np = keep_mask.astype(np.uint8)[None, None, ...]  # Add back the batch and channel dimensions
    filtered_label_5d_tensor = torch.from_numpy(filtered_label_5d_np)

    return filtered_label_5d_tensor, num_lesion_after_filter

@jit(nopython=True)
def filter_components(labeled_array, num_features, min_size):
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
        if component.sum() >= min_size:
            output |= component
    return output