# using simpleITK load mhd and raw files. lets just use that instead of nrrd files
# https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
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

if __name__ == '__main__':

	print('XXXXXXXXXXXXXXXXXXXX')

	array, origin, spacing1 = load_itk('gt_binary.mhd')

	# print(array.shape)
	# maxElement = np.amax(array)
	# print(maxElement)
	# print(origin)
	print(spacing1)


	# array2d = array[0,:,:]
	# print(array2d.shape)
	# for i in range(1,100):
	#     temp_image = array[i,:,:]
	#     array2d = np.append(array2d,temp_image,0)

	# Let's say we have an array img of size m x n x 3 to transform into an array new_img of size 3 x (m*n)
	array2d = array.reshape((array.shape[0]*array.shape[1]), array.shape[2])
	# array2d = new_img.transpose()
	# returning the array back to the original form
	array3d = array2d.reshape(array.shape[0],array.shape[1], array.shape[2])
	print(array3d.shape)


	#implementing marching_cubes algorithm and using trimesh to export as stl- https://forum.image.sc/t/3d-model-from-image-slices/33675/10

	import skimage , trimesh
	from skimage.measure import marching_cubes
	verts, faces, normals, values = marching_cubes(array3d,spacing=spacing1)

	surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
	surf_mesh.export('output.stl')


	# array2d = array2d * 255

	# normalize values so that they are within the range 0-255
	# from keras.utils import normalize
	# array2d=normalize(array2d)

	maxElement1 = np.amax(array2d)
	# print(maxElement1)

	array2d=array2d * 255/maxElement1

	# change to int as that is what is the usual type in png
	array2d = array2d.astype(np.uint8)
	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	# print(array2d)

	from PIL import Image
	im = Image.fromarray(array2d)
	im.save("test6cropped.tiff")
	

