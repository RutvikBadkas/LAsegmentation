# from PIL import Image
# for i in range(0,165):
#     im = Image.open('masklarge'+str(i)+'.tif').convert('L')
#     im = im.resize((256, 256))
#     im.save('mask'+str(i)+'.tif')

# import SimpleITK as sitk
#
# reader = sitk.ImageFileReader()
# reader.SetImageIO("BMPImageIO")
# reader.SetFileName('image.nii')
# image = reader.Execute();
#
#
# # writer = sitk.ImageFileWriter()
# # writer.SetFileName(outputImageFileName)
# # writer.Execute(image)

import os
import numpy as np
import nrrd

# Some sample numpy data
#data = np.zeros((5,4,3,2))
filename = 'test1.nrrd'
testoutput = 'test1out.nrrd'
#datatransposed = data.reshape(-2,c)


image_directory = 'Training Set/'
mask_directory = 'Training Set/'

def load_nrrd(filename):
    readdata, header = nrrd.read(filename)
    print(readdata.shape)
    return readdata

# # Read the data back from file
# readdata, header = nrrd.read(filename)

# # # This converts from (a,b,c) to (c,a,b) so basically shifts c to position 1
# # data = readdata.transpose(2,0,1)

# # This merges (a,b,c) to (a,b*c) hence the 3d array is now a 2d array
# data2d = readdata.reshape(576,-1)

# # data2d is insanely long so gonna crop it to a reasonable size for viewing
# # take the a middle image at z of 40 so 576*40=23040 to 23616 or 24192 if you
# # want to see two images
# # A = data2d[:,23040:23616]
# #
# # from PIL import Image
# # im = Image.fromarray(A)
# # im.save("test1cropped.png")

# # B = readdata[:,:,24]
# # # Try to print the 2d image for viewing
# #
# # from PIL import Image
# # im = Image.fromarray(B)
# # im.save("test2cropped.png")


# datareturned3d = data2d.reshape(576,576,88)

# # Write to a NRRD file
# # nrrd.write(testoutput, datareturned3d)
# readdata2, header2 = nrrd.read(testoutput)
# # check slice to see if it came out in the same order and all
# # C = readdata2[:,:,24]
# # from PIL import Image
# # im = Image.fromarray(C)
# # im.save("test3cropped.png")
# # D=A
# # for i in range(88):
# #     temp_image = readdata[:,:,i]
# #     D = np.append(D,temp_image,1)
# #
# # Dcrop = D[:,23040:23616]
# #
# # from PIL import Image
# # im = Image.fromarray(D)
# # im.save("test4cropped.png")

# print(readdata.shape)
# #print(readdata2.shape)
# #print(data2d.shape)
# print(datareturned3d.shape)
# # print(D.shape) #just checking if B is 2D
# print(header)
# print(header2)



images = os.listdir(image_directory)
j=0
for i, folder_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    # if folder_name == 'data_readme.txt':
    #     continue
    path= image_directory + folder_name
    image_name = '/lgemri.nrrd'
    
    mask_name = '/laendo.nrrd'


    image1= load_nrrd(path+image_name)
    maxElement1 = np.amax(image1)
    #image1= image1 * 255/maxElement1
    
    print(maxElement1)
    

    print('image:' ,image1.shape) 
    mask1= load_nrrd(path+mask_name)
    maxElement3 = np.amax(mask1)
    shape=image1.shape
    
       
    print('mask:' ,mask1.shape)    
    print(maxElement3)

    
    image1 = image1.astype(np.uint8)
    #mask1 = (mask1 > 0)
    mask1 = mask1.astype(np.uint8)
    maxElement3 = np.amax(mask1)
    print(maxElement3)
    # maxElement2 = np.amax(mask1)
    # mask1 * 255/maxElement2
    # print(maxElement2)
    # #print(image_directory+image_name)

    j=j+1
    for i in range(88):
        
        
        temp_image = image1[160:416,160:416,i]
        temp_mask = mask1[160:416,160:416,i]

        maxElement4 = np.amax(temp_mask)
        if maxElement4 == 0:
            continue
        if shape[0] ==640:
            temp_image = image1[192:448,192:448,i]
            temp_mask = mask1[192:448,192:448,i]

        from PIL import Image
        mk = Image.fromarray(temp_mask)
        im = Image.fromarray(temp_image)
        #im.convert('I;16')
        im.save('2Dimages_noblack/image_'+str(j)+'_'+str(i)+'.tif')
        mk.save('2Dmasks_noblack/mask_'+str(j)+'_'+str(i)+'.tif')
