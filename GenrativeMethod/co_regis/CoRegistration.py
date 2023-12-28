from Util_Co import coRegis_fix, itk2sitk, sitk2itk
import matplotlib.pyplot as plt
import SimpleITK as sitk
import json
import os

def plot_and_save(Transfomred_CT, Transformed_MRI, Transformed_label, save_path):
    # Choose the slice you want to display (e.g., slice 0 for the first slice)
    slice_idx = 130

    # Extract the selected slices
    label_slice = Transformed_label[:, :, slice_idx]
    result_slice = Transformed_MRI[:, :, slice_idx]
    ct_slice = Transfomred_CT[:, :, slice_idx]

    # Plot the selected slices
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(sitk.GetArrayViewFromImage(label_slice), cmap='gray')
    plt.title('Label Image')

    plt.subplot(132)
    plt.imshow(sitk.GetArrayViewFromImage(result_slice), cmap='gray')
    plt.title('MRI Image')

    plt.subplot(133)
    plt.imshow(sitk.GetArrayViewFromImage(ct_slice), cmap='gray')
    plt.title('Skull-Stripped Image')

    # Save the plot to a file
    plt.savefig(save_path)

    # Show the plot (optional)
    plt.show()

def main(): 
    
    # get map file between idx_CT and idx_MRI
    map_file = "/home/bruno/xfang/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json"
    with open(map_file, 'r') as file:
        map = json.load(file)
    
    ids =  list(set(['_'.join(filename.split('_')[:2]) for filename in os.listdir("/home/bruno/xfang/dataset/images")]))
    b0_dir = "/scratch4/rsteven1/New_MRI/"
    dwi_dir = "/scratch4/rsteven1/examples"
    label_dir = "/scratch4/rsteven1/examples"
    save_dwi_path = "/scratch4/rsteven1/DWI_coregis_20231208"
    os.makedirs(save_dwi_path ,exist_ok = True)

    for unique_id in ids:

        dwi_id = map.get(unique_id)
        fix_b0_path = os.path.join(b0_dir, unique_id, f'{unique_id}_b0_MNI_coRegist.nii.gz')
        moving_DWI_path = os.path.join(dwi_dir, dwi_id, f'{dwi_id}_DWI_MNI.nii.gz')
        moving_label_path = os.path.join(dwi_dir, dwi_id, f'{dwi_id}_DAGMNet_CH3_Lesion_Predict_MNI.nii.gz')

        fix_b0 = sitk.ReadImage(fix_b0_path)
        moving_DWI = sitk.ReadImage(moving_DWI_path)
        moving_label =sitk.ReadImage(moving_label_path)
    
        label_transformed , DWI_transformed, result_transform_parameters, b0_img= coRegis_fix(fix_b0, moving_DWI, moving_label)

        save_path = f'/home/bruno/xfang/GenrativeMethod/co_regis/save/{unique_id}.png'
        plot_and_save(b0_img, DWI_transformed, label_transformed, save_path)

        DWI_name = f'{unique_id}_DWI_coregis.nii.gz'
        sitk.WriteImage(DWI_transformed, os.path.join(save_dwi_path, DWI_name))
        print('saved one ')
    
    return None

if __name__=="__main__":
    main()