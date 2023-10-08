import os
import torch
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool

class CTDataset(Dataset):
    def __init__(self, CT_image_root, MRI_label_root, transform=None):
        self.CT_path = CT_image_root
        self.MRI_path = MRI_label_root
        self.transform = transform
        self.CT_name = sorted(os.listdir(os.path.join(CT_image_root)))
        self.MRI_name = sorted(os.listdir(os.path.join(MRI_label_root)))
        self.target_size = self.compute_target_size()

    def compute_bounding_box(self, CT_ID):
        CT_image = sitk.ReadImage(os.path.join(self.CT_path, CT_ID), sitk.sitkFloat32)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(CT_image, 0, 1, 200))
        bounding_box = label_shape_filter.GetBoundingBox(1)
        return bounding_box[3:6] 

    def compute_target_size(self):
        with Pool() as pool:
            all_dims = pool.map(self.compute_bounding_box, self.CT_name)
        max_dims = [max(dim) for dim in zip(*all_dims)]  # Find max dimensions across all bounding boxes
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
        CT_ID = self.CT_name[index]
        MRI_ID = self.MRI_name[index]
        CT_image = sitk.ReadImage(os.path.join(self.CT_path, CT_ID), sitk.sitkFloat32)
        MRI_image = sitk.ReadImage(os.path.join(self.MRI_path, MRI_ID), sitk.sitkFloat32)
        
        CT_tensor, MRI_tensor = self.preprocess(CT_image, MRI_image) 
        
        # If you have additional transformations, apply them here
        if self.transform:
            CT_tensor = self.transform(CT_tensor)
            MRI_tensor = self.transform(MRI_tensor)
        
        MRI_one_hot = to_one_hot_3d(MRI_tensor, 2)
        
        return CT_ID, MRI_ID, CT_tensor, MRI_tensor, MRI_one_hot

    def __len__(self):
        return len(self.CT_name)


def to_one_hot_3d(tensor, n_classes): #shape = [batch, s, h, w]-> [batch, s, h, w, c]-> [batch, c, h, w]
    """
    tensor 1 is background prediction output
    tensor 2 is foreward prediction output
    ###### why change it to two tensor??????????????
    """
    b, s, h, w = tensor.size()
    if n_classes == 2:
        tensor1, tensor2 = torch.clone(tensor), torch.clone(tensor)
        tensor1[tensor == 0]  = 1.0
        tensor1[tensor == 1]  = 0.0
        tensor2[tensor == 1]  = 1.0
        tensor2[tensor == 0]  = 0.0
        tensor1, tensor2 = tensor1.unsqueeze(-1), tensor2.unsqueeze(-1)
        one_hot = torch.cat((tensor1, tensor2), -1)
        one_hot = one_hot.squeeze(0)
        one_hot = one_hot.permute(3,0,1,2)
    return one_hot



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