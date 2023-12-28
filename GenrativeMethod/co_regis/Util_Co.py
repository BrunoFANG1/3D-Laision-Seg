#import cv2
import numpy as np
np.set_printoptions(threshold = np.inf)
import scipy
import sys
sys.path.append('../..')
import os
from scipy.ndimage import label
import SimpleITK as sitk
from pathlib import Path
import skimage.io as io
import itk
from shutil import copy

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

def coRegis_fix(target_fixed, target_moving, target_label):
    """
    
    This code is help doing the co-registration between CT-image and MR Image/ MRI lesion label
    We do the registration first of the CT image and MR image. After that, we use the transformation
    matrix to transfer MR label to CT-label
    
    fixed_img: CT Image
    moving_img: MR Image
    moving_label: MR label
    metric_method:
    ["extremity MRI"]: intra-subject; affine transformation, mutual information metric
    ["CTMR-based"]:intra-subject; multi-resolution (4)-
    rigid + B-spline transformation, Mutual Information metric (Mattes) with Adaptive Stochastic Gradient Descent optimizer
    
    We tranfer MR image/ MR label into CT space by initial transform and do the co-registration,
    the MR image and MR label spacing/origin will change and same with CT image
    """
    

    # Generate a centering transform based on the images
    IMAGE_DIMENSION = 3
    fixed_ori, fixed_dir = target_fixed.GetOrigin(), target_fixed.GetDirection()

    # fixed_img = CT_preprocess_skulloff(target_fixed,40, 80, (0.0,1.0),1)
    fixed_img = target_fixed
    fixed_img.SetDirection(fixed_dir)
    fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving_img = sitk.Cast(target_moving, sitk.sitkFloat32)
    label_img = sitk.Cast(target_label, sitk.sitkFloat32)
    print(fixed_img.GetOrigin())
    print(label_img.GetDirection())
    print(moving_img.GetDirection())
    processed_ct  = fixed_img
    print(fixed_img.GetSize())
    print(fixed_img.GetSize())
    print('fixed_ori_direction:',fixed_img.GetDirection())
    print('moving_ori_direction: ',moving_img.GetDirection())

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, # Note these are purposefuly reversed!
        moving_img,# Note these are purposefuly reversed!
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    
    moving_resampled = sitk.Resample(
        moving_img,       # Note these are purposefuly reversed!
        fixed_img,      # Note these are purposefuly reversed!
        initial_transform,
        sitk.sitkNearestNeighbor,# TODO: use different interpolator?
        0.0,              # Note(Jacob): default value
        moving_img.GetPixelID(),
    )
    label_resampled = sitk.Resample(
        label_img,       # Note these are purposefuly reversed!
        fixed_img,      # Note these are purposefuly reversed!
        initial_transform,
        sitk.sitkNearestNeighbor,# TODO: use different interpolator?
        0.0,              # Note(Jacob): default value
        label_img.GetPixelID(),
    )

    itk_fix = sitk2itk(fixed_img,3)
    itk_mov = sitk2itk(moving_resampled,3)
    itk_label = sitk2itk(label_resampled,3)
    parameter_object1 = itk.ParameterObject.New()
    parameter_map_rigid1 = parameter_object1.GetDefaultParameterMap('rigid')
    parameter_map_rigid1['FinalBSplineInterpolationOrder'] = ['0']
    parameter_object1.AddParameterMap(parameter_map_rigid1)
    parameter_map_affine1 = parameter_object1.GetDefaultParameterMap('affine')
    parameter_map_affine1['FinalBSplineInterpolationOrder'] = ['0']
    parameter_object1.AddParameterMap(parameter_map_affine1)

    print('moving to this line')
    result_img, result_transform_parameters = itk.elastix_registration_method(itk_fix,itk_mov, parameter_object = parameter_object1,log_to_console = True)
    
    result_img = itk2sitk(result_img)
    print('moving_result_direction: ',result_img.GetDirection())
    #print('result_img_transfer spacing ',result_img.GetSpacing())
    
    itk_label = sitk2itk(label_resampled,3)
    print('Label_origin origin ', itk_label.GetOrigin())
    print('Label_origin space ', itk_label.GetSpacing())
    
    transform_label_img1 =itk.transformix_filter(itk_label, result_transform_parameters)
    
    #result_img2, result_transform_parameters2 = itk.elastix_registration_method(itk_fix,result_img1, parameter_object = parameter_object2)
    #transform_label_img2 =itk.transformix_filter(transform_label_img1,result_transform_parameters2)

    label_result =itk2sitk(transform_label_img1)
    print('label_result_direction: ',label_result.GetDirection())
    #result_img2 = itk2sitk(result_img2)
    ct_img = itk2sitk(itk_fix)
    ct_img.SetDirection(fixed_dir)
    print('ct_result_direction: ',ct_img.GetDirection())
    print('Label_changed origin ', label_result.GetOrigin())
    print('Label_changed space ', label_result.GetSpacing())
    print('Label_changed size ', label_result.GetSize())

    return label_result , result_img, result_transform_parameters,ct_img

def main():
    print("test coregistration code")
    
    # Load images
    CT_img = sitk.ReadImage("./2224_2564111995_ct.nii")
    MRI_img = sitk.ReadImage("./89baf219_20171117_DWI_MNI.nii")
    label_img =sitk.ReadImage("./89baf219_20171117_DAGMNet_CH3_Lesion_Predict_MNI.nii")

    label_transformed , mri_transformed, result_transform_parameters,ct_img= coRegis_fix(CT_img, MRI_img, label_img)

    # save
    sitk.WriteImage(ct_img,f'2224_2564111995_ct_skullstrip.nii.gz')
    sitk.WriteImage(mri_transformed,f'2224_2564111995_mri_dwi.nii.gz')
    sitk.WriteImage(label_transformed,f'2224_2564111995_label.nii.gz')


if __name__=="__main__":
    main()
