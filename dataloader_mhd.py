# using simpleITK load mhd and raw files. My dataset had mhd/raw hence had to make this function but if your files are .nii or .dicom then you can import super easily with a single line.

import skimage , trimesh
import SimpleITK as sitk
import numpy as np
from PIL import Image
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(itkimage.GetOrigin()))

    # Read the spacing along each dimension
    spacing = np.array(list(itkimage.GetSpacing()))

    return ct_scan, origin, spacing

