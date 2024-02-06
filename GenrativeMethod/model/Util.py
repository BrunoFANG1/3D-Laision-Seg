import monai
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, 
    SpatialPadd, RandSpatialCropd, RandAffined, ToTensord,
    RandRotated, RandZoomd, RandFlipd
)

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
                                                # rotate_range=(360, 360, 360),
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
        return monai.networks.nets.AttentionUnet(
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
        print('we are using Swin_UnetR model')
        return monai.networks.nets.SwinUNETR(img_size=(128, 128,128), in_channels=1, out_channels=1, feature_size=24)
    
    elif args.model_name == 'Unet++':
        print('we are using Unet++ model')
        return monai.networks.nets.BasicUNetPlusPlus(spatial_dims = 3, 
                              out_channels = 1,
                              features=(32, 32, 64, 128, 256, 32),
                              deep_supervision = True
                              )
    else:
        assert False