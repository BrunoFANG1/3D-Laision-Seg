import os
import torch
import SimpleITK as sitk
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torchvision.transforms.functional as TF
import random
import torchio as tio
from Util import random_crop_around_lesion
random.seed(0)  # You can choose any number as the seed value

class StrokeAI(Dataset):
    def __init__(self, 
                 CT_root, DWI_root, ADC_root, label_root, MRI_type, 
                 mode = 'train',
                 map_file=None, bounding_box=False, indices=None, 
                 transform=None, 
                 padding = False, slicing=False, instance_normalize=False, crop = False, RotatingResize = False):
        """
        CT_root, MRI_root, label_root: The path to CTs, MRIs, segmentation labels.
        indices: Don't know what it is.
        transform: image transformation.
        padding: The orginal input is in the form of (189, 233, 197) to equal length (a, a, a) e.g. (223, 223, 223).
        slicing: Convert 3D image to a 2D slice.
        normalize: preprocessing, make the dataset zero mean and unit variance.
        crop: crop to a certain cubic that must contain lesion eg. (56,56,56)
        """
        self.ct_dir = CT_root
        self.dwi_dir = DWI_root
        self.adc_dir = ADC_root
        self.label_dir = label_root
        self.MRI_type = MRI_type
        self.mode = mode
        self.ids = self.get_unique_ids()

        # Due to Rongxi's terrible naming system, I have to do that
        if map_file is not None:
            with open(map_file, 'r') as file:
                self.ct_map_mri = json.load(file)

        ### adjustment choice
        self.bounding = bounding_box
        self.transform = transform
        self.slicing = slicing
        self.instance_normalize = instance_normalize
        self.padding = padding
        self.cropping = crop
        self.RotatingResize = RotatingResize
        if self.RotatingResize: 
            print("Rotation and Resize")

        if self.slicing == True:
            print("we convert 3D to 2D images along axiel")
            self.slicing_num = 16
            print("The output size is (190, 190, 16)")

    def get_unique_ids(self):
        # Extract unique IDs from the CT directory
        ids =  ['_'.join(filename.split('_')[:2]) for filename in os.listdir(self.ct_dir)]
        return list(set(ids))
    
    def load_sitk_file(self, file_path):
        # Load a file using SimpleITK
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)

    def __len__(self):
        return len(self.ids)
   
    def preprocess(self, ct_sitk, mri_sitk, label_sitk):
        '''
        Input:  
        CT in sitk form 
        MRI in sitk form
        label in sitk form
        
        process step: bounding, padding, normalize

        Oputput: 
        processed CT in tensor form
        processed MRI in tensor form
        processed label in tensor form
        '''

        # load from sitk and get numpy format
        if self.bounding:
            target_dim = [156, 192, 162]
            cropped_CT = sitk.RegionOfInterest(ct_sitk, target_dim, start_index)
            cropped_MRI = sitk.RegionOfInterest(mri_sitk, target_dim, start_index)
            cropped_label = sitk.RegionOfInterest(label_sitk, target_dim, start_index)

            # Convert the cropped SimpleITK images to NumPy arrays
            ct_array = sitk.GetArrayFromImage(cropped_CT)
            mri_array = sitk.GetArrayFromImage(cropped_MRI)
            label_array = sitk.GetArrayFromImage(cropped_label)
        else:
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            mri_array = sitk.GetArrayFromImage(mri_sitk)
            label_array = sitk.GetArrayFromImage(label_sitk)

        if self.padding:
            ct_array = ct_array[:-1,:-1,:-1]
            mri_array = mri_array[:-1,:-1,:-1]
            label_array = label_array[:-1,:-1,:-1]

        if self.slicing:
            ct_array = ct_array[:,:,90:(90+self.slicing_num)] 
            mri_array = mri_array[:,:,90:(90+self.slicing_num)]
            label_array = label_array[:,:,90:(90+self.slicing_num)]

        # add one dim as channel
        ct_array = np.expand_dims(ct_array, axis=0)
        mri_array = np.expand_dims(mri_array, axis=0)
        label_array = np.expand_dims(label_array, axis=0)

        if self.RotatingResize:

            # define Rotation and resize
            rotation_angles = np.random.uniform(-8, 8, size=3)
            scaling_factors = np.random.uniform(0.8, 1.2, size=3)
            image_affine_transformation = tio.Affine(
                scales=scaling_factors,
                degrees=rotation_angles,
                translation = (0, 0, 0),
                image_interpolation='linear'
            )
            label_affine_transformation = tio.Affine(
                scales=scaling_factors,
                degrees=rotation_angles,
                translation = (0, 0, 0),
                image_interpolation='nearest'
            )
            image_transformation_pipeline = tio.Compose([image_affine_transformation])
            label_transformation_pipeline = tio.Compose([label_affine_transformation])

            # convert array to tio subject
            ct_image = tio.ScalarImage(tensor=ct_array)
            mri_image = tio.ScalarImage(tensor=mri_array)
            label_image = tio.ScalarImage(tensor=label_array)
            image_subject = tio.Subject(
                                ct=ct_image,
                                mri=mri_image,
                            )
            label_subject = tio.Subject(label=label_image)

            image_transformed_subject = image_transformation_pipeline(image_subject)
            label_trnasformed_subject = label_transformation_pipeline(label_subject)

            ct_array = (image_transformed_subject.ct).numpy()
            mri_array = (image_transformed_subject.mri).numpy()
            label_array = (label_trnasformed_subject.label).numpy()
            # 4-dim

        if self.cropping: # sometime it does not work and return a all zeros label
            assert self.slicing == False

            # Find a random point that is lesion (We have already make sure each img has at least one lesion point)
            leision_indices = np.argwhere(label_array == 1)
            if len(leision_indices) > 0:
                random_leision_index = tuple(random.choice(leision_indices))
            else:
                assert False
        
            ct_array, mri_array, label_array = random_crop_around_lesion(ct_array, mri_array, label_array,  random_leision_index, crop_size=(56, 56, 56))


        # convert numpy to torch tensor
        ct_tensor = torch.tensor(ct_array)
        mri_tensor = torch.tensor(mri_array)
        label_array = label_array.astype(np.float32)
        label_tensor = torch.tensor(label_array)

        # if time permits, add them to preprocess function
        # Normalize if required, One question here: Should I normalize the data? Becuase it seems really dark
        if self.instance_normalize:
            ct_instance_norm = torch.nn.InstanceNorm3d(num_features=1, affine=True)
            mri_instance_norm = torch.nn.InstanceNorm3d(num_features=1, affine=True)

            ct_tensor = ct_instance_norm(ct_tensor).detach()
            mri_tensor = mri_instance_norm(mri_tensor).detach()
            # print("MRI Data After Normalization - Mean:", torch.mean(mri_tensor), "Std Dev:", torch.std(mri_tensor))

        return ct_tensor, mri_tensor, label_tensor
       

    def __getitem__(self, idx):
        # read ct, mri, label and add a channel dimension
        unique_id = self.ids[idx]
        # mri_id = self.ct_map_mri.get(unique_id)
        
        # locate CT, MRI, Label
        ct_path = os.path.join(self.ct_dir, f'{unique_id}_ct.nii.gz')
        if self.MRI_type == 'DWI':
            mri_path = os.path.join(self.dwi_dir, f'{unique_id}_DWI_coregis.nii.gz')
        elif self.MRI_type == 'ADC':
            mri_path = os.path.join(self.adc_dir, f'{unique_id}_ADC_coregis.nii.gz')   
        else:
            assert False
        label_path = os.path.join(self.label_dir, f'{unique_id}_label_inskull.nii.gz')

        # read file in sitk format
        ct_sitk = sitk.ReadImage(ct_path)
        mri_sitk = sitk.ReadImage(mri_path)
        label_sitk = sitk.ReadImage(label_path)

        # Preprocess
        if self.mode == 'train':
            ct_tensor, mri_tensor, label_tensor = self.preprocess(ct_sitk, mri_sitk, label_sitk)

        if self.mode == 'test':
            ct_np = sitk.GetArrayFromImage(ct_sitk)
            mri_np = sitk.GetArrayFromImage(mri_sitk)
            label_np = sitk.GetArrayFromImage(label_sitk).astype(np.int32)

            # Convert NumPy arrays to PyTorch tensors
            ct_tensor = torch.from_numpy(ct_np).float()
            mri_tensor = torch.from_numpy(mri_np).float()
            label_tensor = torch.from_numpy(label_np).float()

            ct_tensor = ct_tensor.unsqueeze(0)  # Add a batch dimension
            mri_tensor = mri_tensor.unsqueeze(0)
            label_tensor = label_tensor.unsqueeze(0)

        sample = {'ct': ct_tensor, 'mri': mri_tensor, 'label': label_tensor}
        return sample

def plot_and_save(Transfomred_CT, Transformed_MRI, Transformed_label, save_path):
    # Choose the slice you want to display (e.g., slice 0 for the first slice)
    slice_idx = 26

    # Extract the selected slices
    label_slice = Transformed_label[:, :, slice_idx]
    result_slice = Transformed_MRI[:, :, slice_idx]
    ct_slice = Transfomred_CT[:, :, slice_idx]

    # Plot the selected slices
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(label_slice, cmap='gray')
    plt.title('Label Image')

    plt.subplot(132)
    plt.imshow(result_slice, cmap='gray')
    plt.title('MRI Image')

    plt.subplot(133)
    plt.imshow(ct_slice, cmap='gray')
    plt.title('CT slice')

    # Save the plot to a file
    plt.savefig(save_path)

    # Show the plot (optional)
    plt.show()

# Example code to calculate mean and std of the entire dataset
def calculate_global_mean_std(dataset):
    sum_ct, sum_squared_ct, num_samples = 0.0, 0.0, 0
    sum_mri, sum_squared_mri, num_samples = 0.0, 0.0, 0
    for data in dataset:
        ct_tensor = data['ct']
        mri_tensor = data['mri']  # Assuming data['mri'] is your MRI tensor

        sum_ct += torch.mean(ct_tensor)
        sum_mri += torch.mean(mri_tensor)
        sum_squared_ct += torch.mean(ct_tensor ** 2)
        sum_squared_mri += torch.mean(mri_tensor ** 2)
        num_samples += 1

    mean_ct = sum_ct / num_samples
    mean_mri = sum_mri / num_samples
    std_ct = (sum_squared_ct / num_samples - mean_ct ** 2) ** 0.5
    std_mri = (sum_squared_mri / num_samples - mean_mri ** 2) ** 0.5

    return mean_ct.item(), std_ct.item(), mean_mri.item(), std_mri.item()



def main():
    print("start working")
    train_set = StrokeAI(CT_root="/home/bruno/xfang/dataset/images",
                       DWI_root="/scratch4/rsteven1/DWI_coregis_20231208",  #DWI
                       ADC_root="/scratch4/rsteven1/ADC_coregis_20231228",  # ADC
                       label_root="/home/bruno/xfang/dataset/labels", 
                       MRI_type = 'ADC',
                       mode = 'test',
                       map_file= "/home/bruno/3D-Laision-Seg/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json",
                    #    bounding_box=False,
                       instance_normalize=True, 
                    #    padding=False, 
                    #    slicing=True,
                       crop=False,
                       RotatingResize = False)
    
    print(f"dataset length is {len(train_set)}")
    print("data loads fine")

    # uncomment the following line to calculate mean and var under different conditions, why the result is diffrent for each run
    # ct_mean, ct_std, mri_mean, mri_std = calculate_global_mean_std(train_set)
    # print("CT Mean:", ct_mean)
    # print("CT Std Dev:", ct_std)
    # print("MRI Mean:", mri_mean)
    # print("MRI Std Dev:", mri_std)
    
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    print("data loader fine")

    # Use the function to calculate mean and std

    for batch in train_loader:
        ct_batch = batch['ct']
        mri_batch = batch['mri']
        label_batch = batch['label']

        import pdb
        pdb.set_trace()

        # Calculate the mean and standard deviation of the batch
        ct_mean = ct_batch.mean()
        ct_std = ct_batch.std()
        mri_mean = mri_batch.mean()
        mri_std = mri_batch.std()
        label_mean = label_batch.mean()
        label_std = label_batch.std()

        # print("CT Mean:", ct_mean)
        # print("CT Std Dev:", ct_std)
        # print("MRI Mean:", mri_mean)
        # print("MRI Std Dev:", mri_std)
        # print("Label Mean:", label_mean)
        # print("Label Std Dev:", label_std)
        
        # print(torch.sum(label_batch))

        plot_and_save(ct_batch[0,0], mri_batch[0,0], label_batch[0,0], "./test.png")
        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    main()