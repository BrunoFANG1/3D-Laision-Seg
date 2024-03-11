import torch

import cv2
import numpy as np
np.set_printoptions(threshold = np.inf)
import pydicom
import matplotlib.pyplot as plt
import dicom2nifti
import nibabel as nib
import scipy
import sys
from sklearn import preprocessing
from skimage import exposure
import os
import pandas as pd
from scipy.ndimage import label
from scipy.ndimage.morphology import generate_binary_structure
import SimpleITK as sitk
from pathlib import Path
import skimage.morphology as sm
import glob
from shutil import copy
import pandas as pd
import itk
import itkwidgets

def CT_preprocess_final(nif_dat_path, new_spacing ,level, window, scale, fltsize, Origin):
    """ Given an 3D nifti data, scale into window and level. Scale between
    expects a tuple (new_min, new_max) that determines the new range.The preprocess 
    also do the resampling and registrationof the CT image.
    Works with both 2D and 3D data.
    
    nif_dat: input the data from sitkImage Read
    new_spacing: the voxel size: [voxel[0], voxel[1], voxel[2]]
    level : center of window, window = width of window.
    scale : (min_scale, max_scale) to rescale in-window values to.
    filt_size : the median filter kernal size; Ex:fltsize = 5  = median_filter(5,5,5)
    Origin: The nifti data origin

    """
    
    MNI_152 = sitk.ReadImage(r"/home/bruno/test/mni_icbm152_t1_tal_nlin_asym_09a.nii") 
    nif_dat = sitk.ReadImage(nif_dat_path) 

    fixed_img = sitk.Cast(MNI_152, sitk.sitkFloat32)
    moving_img = sitk.Cast(nif_dat, sitk.sitkFloat32)
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, # Note these are purposefuly reversed!
        moving_img,# Note these are purposefuly reversed!
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(
        moving_img,       # Note these are purposefuly reversed!
        fixed_img,      # Note these are purposefuly reversed!
        initial_transform,
        sitk.sitkLinear,  # TODO: use different interpolator?
        0.0,              # Note(Jacob): default value
        moving_img.GetPixelID(),
    )
    npmoving = sitk.GetArrayFromImage(moving_resampled)
    plt.figure(1)
    plt.imshow(npmoving[80,:,:],cmap ='gray')
    plt.colorbar()
    itk_fix = sitk2itk(fixed_img,3)
    itk_mov = sitk2itk(moving_resampled,3)
    print(itk.size(itk_mov))
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)
    result_img, result_transform_parameters = itk.elastix_registration_method(itk_fix,itk_mov, parameter_object = parameter_object,log_to_console = True)
    result_img = itk2sitk(result_img)
    print(result_img.GetSize())
    print(result_img.GetSize())
    npmoving = sitk.GetArrayFromImage(result_img)
    plt.figure(2)
    plt.imshow(npmoving[80,:,:],cmap ='gray')
    plt.colorbar()
    print(npmoving[80,:,:].shape)
    # denoise the image by 3D median filter
    npmoving = scipy.ndimage.median_filter(npmoving, size = (fltsize,fltsize,fltsize))    
    # Extract the brain
    im_size = npmoving.shape
    print(im_size)
    upper = level + window/2.0
    lower = level - window/2.0
    npmoving[npmoving > upper] = upper
    npmoving[npmoving < lower] = lower
    plt.figure(3)
    plt.imshow(npmoving[80,:,:],cmap ='gray')
    plt.colorbar()
    
    re_img = np.zeros(im_size)
#    upper = level + window/2.0
#   lower = level - window/2.0
    for num in range(im_size[0]):
        img1 = npmoving[num, :, :]
        new_img = img1.copy()        
        new_img[new_img > upper] = upper
        new_img[new_img < lower] = lower

    # Delete the Gantry
        if new_img.max() > 0: 
            ref, bi_img = cv2.threshold(img1 ,0,1,cv2.THRESH_BINARY)
            bi_img = np.uint8(bi_img)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bi_img)
            op = np.zeros(new_img.shape)
            pixel_label = np.resize(labels, labels.shape[0]*labels.shape[1])            
            num1 = np.delete(np.bincount(pixel_label),0)
            label_brain = np.where(num1 == num1.max())[0]+1
            mask = labels == label_brain
            brain_pos = np.where(mask[:,:] == True)
            op[mask] = new_img[brain_pos]
            op.astype(float)
            re_img[num, :, :] = op
        elif new_img.max() == 0:
            op = new_img
            re_img[num, :, :] = op
    re_img = (re_img -lower)/ (upper-lower) * (scale[1] - scale[0])
    re_img = sitk.GetImageFromArray(re_img)

    print('re_img size: ', re_img.GetSize())
    print('re_img spacing: ', re_img.GetSpacing())
   
    return re_img

def sitk2itk(sitk_img, IMAGE_DIMENSION):
    npsitk_img = sitk.GetArrayFromImage(sitk_img)
    itk_img = itk.GetImageFromArray(npsitk_img)
    itk_img.SetOrigin(sitk_img.GetOrigin())
    itk_img.SetSpacing(sitk_img.GetSpacing())
    itk_img.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(sitk_img.GetDirection()), [IMAGE_DIMENSION]*2)))
    return itk_img
def itk2sitk(itk_img):
    npitk_img = itk.GetArrayFromImage(itk_img)
    sitk_img = sitk.GetImageFromArray(npitk_img, isVector = itk_img.GetNumberOfComponentsPerPixel()>1)
    sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    return sitk_img

path = r'/home/bruno/test/2.25.242520136652711996329853921667886633040_01_Trauma_Head_Neuro_20230107035450_7.nii'
img2 = CT_preprocess_final(path, [1.0, 1.0, 1.0] ,40, 80, (0.0,1.0),1,(0,0,0))
img2_new = sitk.GetArrayFromImage(img2)

sitk.WriteImage(img2, './test_1.nii')
print("finish")