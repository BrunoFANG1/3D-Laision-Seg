import os
import json
import ants
from multiprocessing import Pool
import torch

def register_pair(pair_info):
    patient_id, private_patient_id = pair_info
    
    # Load the CT image
    ct_image_path = f"/home/bruno/xfang/dataset/images/{patient_id}_ct.nii.gz"
    if not os.path.exists(ct_image_path):
        print(f"CT image not found for patient {patient_id}")
        return None
    ct_image = ants.image_read(ct_image_path)
    
    # Load the MRI image
    mri_image_path = f"/scratch4/rsteven1/StrokeAI/examples/{private_patient_id}/{private_patient_id}_DWI_MNI.nii.gz"
    if not os.path.exists(mri_image_path):
        print(f"MRI image not found for patient {patient_id}")
        return patient_id
    mri_image = ants.image_read(mri_image_path)
    
    # Load the lesion label
    lesion_label_path = f"/scratch4/rsteven1/StrokeAI/examples/{private_patient_id}/{private_patient_id}_DAGMNet_CH3_Lesion_Predict_MNI.nii.gz"
    if not os.path.exists(lesion_label_path):
        print(f"Lesion label not found for patient {patient_id}")
        return patient_id
    lesion_label = ants.image_read(lesion_label_path)
    
    # Perform intensity normalization
    # normalized_mri = ants.registration.histogram_match(mri_image, ct_image)
    
    # Perform registration with GPU acceleration
    syn_transform = ants.registration(
        fixed=ct_image,
        moving=mri_image,
        type_of_transform='SyN',
        syn_metric='MI',
        syn_sampling=32,
        reg_iterations=(40, 20, 0),
        verbose=False,
        use_cuda=True
    )
    
    # Apply the transformation to the MRI image
    registered_mri = ants.apply_transforms(
        fixed=ct_image,
        moving=mri_image,
        transformlist=syn_transform['fwdtransforms']
    )
    
    # Apply the same transformation to the lesion label
    registered_label = ants.apply_transforms(
        fixed=ct_image,
        moving=lesion_label,
        transformlist=syn_transform['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    
    # Save the registered MRI image and label
    output_dir = f"./{patient_id}"
    os.makedirs(output_dir, exist_ok=True)
    ants.image_write(registered_mri, f"{output_dir}/{patient_id}_registered_mri.nii.gz")
    ants.image_write(registered_label, f"{output_dir}/{patient_id}_registered_label.nii.gz")
    
    return patient_id


# Set the number of processes to match the number of available CPUs
num_processes = os.cpu_count()

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()

print(f"Number of available CPUs: {num_processes}")
print(f"Number of available GPUs: {num_gpus}")

# Load the JSON file containing the patient ID mappings
with open("/home/bruno/3D-Laision-Seg/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json", "r") as f:
    patient_id_mapping = json.load(f)

# Get the list of patient IDs from the CT file names
ct_files = [f for f in os.listdir("/home/bruno/xfang/dataset/images") if f.endswith("_ct.nii.gz")]
patient_ids = ['_'.join(f.split('_')[:2]) for f in ct_files]

# Create a list of image pair information
image_pairs_info = [(patient_id, patient_id_mapping[patient_id]) for patient_id in patient_ids]

# Create a pool of processes
pool = Pool(processes=num_processes)

# Register image pairs in parallel and track progress
num_pairs = len(image_pairs_info)
completed_pairs = 0
missing_files_patients = []

for patient_id in pool.imap_unordered(register_pair, image_pairs_info):
    if patient_id is not None:
        completed_pairs += 1
        print(f"Completed registration for patient {patient_id} ({completed_pairs}/{num_pairs})")
    else:
        missing_files_patients.append(patient_id)

pool.close()
pool.join()

print("Registration completed for all pairs.")

# Save the patient IDs with missing files to a text file
with open("missing_files_patients.txt", "w") as f:
    for patient_id in missing_files_patients:
        f.write(f"{patient_id}\n")

print("Patient IDs with missing files are saved in 'missing_files_patients.txt'.")