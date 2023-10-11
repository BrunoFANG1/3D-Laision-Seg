import os
import torch
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool
import os
import json

class CTDataset(Dataset):
    def __init__(self, CT_image_root, MRI_label_root, transform=None):
        self.CT_path = CT_image_root
        self.MRI_path = MRI_label_root
        self.transform = transform
        self.CT_name = sorted(os.listdir(os.path.join(CT_image_root)))
        self.MRI_name = sorted(os.listdir(os.path.join(MRI_label_root)))
        
        if os.path.exists('max_dims.json'):
            with open('max_dims.json', 'r') as f:
                self.target_size = tuple(json.load(f))
        else:
            self.target_size = self.compute_target_size()

    def compute_target_size(self):
         # Find max dimensions across all bounding boxes
        
        if os.path.exists('max_dims.json'):
            with open('max_dims.json', 'w') as f:
                json.dump(max_dims, f)
        else:
            print("failure loading bounding box dim")
            break

        return tuple(max_dims)

    def preprocess(self, CT_image, MRI_image):
        # Get the bounding box for CT image
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(CT_image, 0, 1, 200))
        bounding_box = label_shape_filter.GetBoundingBox(1)
        
        # Crop both CT and MRI images using the same bounding box
        cropped_CT = sitk.RegionOfInterest(CT_image, bounding_box[3:6], bounding_box[0:3])
        cropped_MRI = sitk.RegionOfInterest(MRI_image, bounding_box[3:6], bounding_box[0:3])
        
        # Convert to numpy arrays
        CT_array = sitk.GetArrayFromImage(cropped_CT)
        MRI_array = sitk.GetArrayFromImage(cropped_MRI)
        
        # Convert to torch tensors
        CT_tensor = torch.FloatTensor(CT_array).unsqueeze(0)  # Adding channel dimension
        MRI_tensor = torch.FloatTensor(MRI_array).unsqueeze(0)  # Adding channel dimension
        
        return CT_tensor, MRI_tensor

    def __getitem__(self, index):
        ############################# We have a bounding box, make sure brain is not larger than bounding box or we need to resize the image
        
        CT_ID = self.CT_name[index]
        MRI_ID = self.MRI_name[index]

        CT_image = sitk.ReadImage(os.path.join(self.CT_path, CT_ID), sitk.sitkFloat32)
        MRI_image = sitk.ReadImage(os.path.join(self.MRI_path, MRI_ID), sitk.sitkFloat32)
        
        CT_tensor, MRI_tensor = self.preprocess(CT_image, MRI_image) 
        
        # If you have additional transformations, apply them here
        if self.transform:
            CT_tensor = self.transform(CT_tensor)
            MRI_tensor = self.transform(MRI_tensor)
        
        
        return CT_ID, MRI_ID, CT_tensor, MRI_tensor

    def __len__(self):
        return len(self.CT_name)


def main():
    
    print("start working")
    train_set = CTDataset(CT_image_root = "../dataset/images/", MRI_label_root = "../dataset/labels/")
    print(f"dataset length is {len(train_set)}")
    print("data loads fine")
    train_loader = DataLoader(dataset = train_set, batch_size = 1, shuffle=True)
    for CT_ID, MRI_ID, CT_preprocess, MRI_preprocess, MRI_preprocess1 in train_loader:
        print(f"The ct id is {CT_ID}")
        print(f"The mri id is {MRI_ID}")
        print(f"The shape of MRI preprocess is {CT_preprocess.shape}")
        print(f"max of image is {CT_preprocess.max()}")
        print(f"min of image is {CT_preprocess.min()}")
        print(f"mean is {CT_preprocess.mean()}")
        print(f"ratio of zero and not {torch.sum(CT_preprocess==0)/torch.sum(CT_preprocess!=0)}")
        print(f"max is {MRI_preprocess.max()}")
        print(f"min is {MRI_preprocess.min()}")
        print(f"label shape is {MRI_preprocess1.shape}")
        print(f"label max is {MRI_preprocess1.max()}")
        break


if __name__ == "__main__":
    main()