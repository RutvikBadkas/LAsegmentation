import skimage , trimesh
from skimage.measure import marching_cubes
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
import os
from dataloader_mhd import load_itk

TEST_DIR = "Tests3d/"
TEST_DIR_FOLDERS = os.listdir(TEST_DIR)
FILENAME = "/prediction_reg.nii"
MASK_DIR = "data/evaluate/"

for i, folder_name in enumerate(TEST_DIR_FOLDERS):    #Remember enumerate method adds a counter and returns the enumerate object
	
	if folder_name == 'data_readme.txt':
		continue

	path= TEST_DIR + folder_name

	try:
		img = nib.load(path+FILENAME)

		a = np.array(img.dataobj)

		control, origin, spacing1 = load_itk(MASK_DIR+'image.mhd')
		verts1, faces1, normals1, values1 = marching_cubes(control,spacing=spacing1)

		verts, faces, normals, values = marching_cubes(a,spacing=spacing1)

		surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
		surf_mesh.export(path+'/mesh_reg.stl')

		v=np.array(verts1) # control vertices
		u=np.array(verts) # predicted vertices

		haus=directed_hausdorff(u, v)[0]
		print(haus)

		f=open(path+"/Hausdorff_reg.txt",'w')
		f.write(str(haus))
		f.close()
		
	except Exception as e:
		print("error occured in: "+folder_name)
		print(e)
		continue