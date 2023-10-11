import os
import json
from multiprocessing import Pool
import SimpleITK as sitk

def compute_bounding_box(CT_path, CT_ID):
    CT_image = sitk.ReadImage(os.path.join(CT_path, CT_ID), sitk.sitkFloat32)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(CT_image, 0, 1, 200))
    bounding_box = label_shape_filter.GetBoundingBox(1)
    return bounding_box[3:6] 

def wrapped_compute_bounding_box(args):
    CT_path, CT_ID, index, total = args
    print(f"Processing image {index + 1}/{total}")
    return compute_bounding_box(CT_path, CT_ID)

def compute_target_size(CT_path, CT_name_list):
    total_images = len(CT_name_list)
    print(f"Total CT images to process: {total_images}")

    all_dims = []
    for index, CT_ID in enumerate(CT_name_list):
        print(f"Processing image {index + 1}/{total_images}")
        dims = compute_bounding_box(CT_path, CT_ID)
        all_dims.append(dims)

    max_dims = [max(dim) for dim in zip(*all_dims)]

    with open('max_dims.json', 'w') as f:
        json.dump(max_dims, f)
        
    return tuple(max_dims)


def main():
    CT_image_root = "../dataset/images/"
    CT_name_list = sorted(os.listdir(CT_image_root))
    
    if os.path.exists('max_dims.json'):
        with open('max_dims.json', 'r') as f:
            target_size = tuple(json.load(f))
        print(f"Loaded max_dims from file: {target_size}")
    else:
        target_size = compute_target_size(CT_image_root, CT_name_list)
        print(f"Computed and saved max_dims: {target_size}")

if __name__ == "__main__":
    main()
