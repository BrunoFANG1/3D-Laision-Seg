import numpy as np
import random

# def random_crop_around_lesion(ct_array, mri_array, label_array, lesion_index, crop_size=(56, 56, 56)):
#     """
#     Return a random crop of the specified size from the CT, MRI, and label arrays that contains the lesion point.
#     Assumes all arrays are of the same dimensions.

#     :param ct_array: A 3D numpy array for CT
#     :param mri_array: A 3D numpy array for MRI
#     :param label_array: A 3D numpy array for labels
#     :param lesion_index: A tuple (x, y, z) representing the lesion point coordinates
#     :param crop_size: Desired crop size as a tuple (default is (56, 56, 56))
#     :return: Cropped CT, MRI, and label numpy arrays
#     """
#     image_size = ct_array.shape

#     # Calculate the range for the starting point of the crop
#     start_range = [max(0, lesion_index[i] - crop_size[i]) for i in range(3)]
#     end_range = [min(image_size[i] - crop_size[i], lesion_index[i] + crop_size[i]//2) for i in range(3)]

#     # Adjust the range to ensure the crop is within the image boundaries
#     start_point = [random.randint(start_range[i], min(end_range[i], image_size[i] - crop_size[i])) for i in range(3)]

#     # Define end points for cropping
#     end_point = [start_point[i] + crop_size[i] for i in range(3)]

#     # Crop the arrays
#     cropped_ct = ct_array[start_point[0]:end_point[0], start_point[1]:end_point[1], start_point[2]:end_point[2]]
#     cropped_mri = mri_array[start_point[0]:end_point[0], start_point[1]:end_point[1], start_point[2]:end_point[2]]
#     cropped_label = label_array[start_point[0]:end_point[0], start_point[1]:end_point[1], start_point[2]:end_point[2]]

#     return cropped_ct, cropped_mri, cropped_label


def random_crop_around_lesion(ct_array, mri_array, label_array, lesion_index, crop_size=(56, 56, 56)):
    """
    Return a random crop of the specified size from the CT, MRI, and label arrays that contains the lesion point.
    Assumes all arrays are of the same dimensions.

    :param ct_array: A 3D numpy array for CT
    :param mri_array: A 3D numpy array for MRI
    :param label_array: A 3D numpy array for labels
    :param lesion_index: A tuple (x, y, z) representing the lesion point coordinates
    :param crop_size: Desired crop size as a tuple (default is (56, 56, 56))
    :return: Cropped CT, MRI, and label numpy arrays
    """
    _, x, y, z = lesion_index
    cx, cy, cz = crop_size

    # Ensure crop size is not larger than the array size
    assert cx < ct_array.shape[1]
    assert cy < ct_array.shape[2]
    assert cz < ct_array.shape[3]

    # Calculate the range for the starting point of the crop
    x_min = max(0, x - cx + 1)
    x_max = min(ct_array.shape[1] - cx, x)
    y_min = max(0, y - cy + 1)
    y_max = min(ct_array.shape[2] - cy, y)
    z_min = max(0, z - cz + 1)
    z_max = min(ct_array.shape[3] - cz, z)

    # Select a random starting point within the range
    start_x = np.random.randint(x_min, x_max)
    start_y = np.random.randint(y_min, y_max)
    start_z = np.random.randint(z_min, z_max)

    # Crop the arrays
    cropped_ct = ct_array[:, start_x:start_x + cx, start_y:start_y + cy, start_z:start_z + cz]
    cropped_mri = mri_array[:, start_x:start_x + cx, start_y:start_y + cy, start_z:start_z + cz]
    cropped_label = label_array[:, start_x:start_x + cx, start_y:start_y + cy, start_z:start_z + cz]

    return cropped_ct.astype(np.float32), cropped_mri.astype(np.float32), cropped_label.astype(np.float32)

# Example usage:
# cropped_ct, cropped_mri, cropped_label = random_crop_around_lesion(ct_array, mri_array, label_array, lesion_index)
