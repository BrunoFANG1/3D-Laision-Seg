import csv
from skimage.measure import label
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage
from scipy.stats import describe
from torch.utils.data import DataLoader
import numpy as np
from CTDataset import CTDataset  # Assuming CTDataset is in a file named CTDataset.py

def main():
    print("start working")
    train_set = CTDataset(CT_image_root="../dataset/images/", MRI_label_root="../dataset/labels/", padding=False, slicing=False)
    print(f"dataset length is {len(train_set)}")
    print("data loads fine")
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    
    with open('analysis_wo_normalize.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header to the CSV file
        writer.writerow(['CT_ID', 'MRI_ID', 'Num_Lesions', 'Lesion_ID', 'Lesion_Size', 'Lesion_Mean', 'Lesion_Variance', 'Surrounding_Mean', 'Surrounding_Variance'])

        for CT_ID, MRI_ID, CT_preprocess, MRI_preprocess in train_loader:
            print(CT_ID)

            MRI_numpy = MRI_preprocess.squeeze().numpy()
            
            # Identify connected regions with label = 1
            labeled_lesions = label(MRI_numpy == 1, connectivity=3)

            # Remove small objects
            min_size = 10  # Set the minimum size of lesions to keep
            filtered_lesions = remove_small_objects(labeled_lesions > 0, min_size=min_size, connectivity=3)
            labeled_lesions, num_lesions = label(filtered_lesions, connectivity=3, return_num=True)
            
            for lesion_id in range(1, num_lesions + 1):
                lesion_mask = labeled_lesions == lesion_id
                lesion_size = np.sum(lesion_mask)
                lesion_data = CT_preprocess.squeeze()[lesion_mask].numpy()
                lesion_desc = describe(lesion_data)
                
                dilated_mask = ndimage.binary_dilation(lesion_mask, iterations=2)
                surrounding_mask = dilated_mask & ~lesion_mask
                surrounding_data = CT_preprocess.squeeze()[surrounding_mask].numpy()
                surrounding_desc = describe(surrounding_data)
                
                # Write the data for each lesion to the CSV file
                writer.writerow([CT_ID[0], MRI_ID[0], num_lesions, lesion_id, lesion_size, lesion_desc.mean, lesion_desc.variance, surrounding_desc.mean, surrounding_desc.variance])

if __name__ == "__main__":
    main()
