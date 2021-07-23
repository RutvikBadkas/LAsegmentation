# This is the MAIN file. Run this and the magic will begin. This will automatically generate a sub folder in /tests3d to store all the files related to the run with a timestamp.
# This file outputs, the model, a prediction segmentation in .nii format made from the data in the eval folder, an excel sheet with the timestamps of the epochs and evolution
# of the loss functions, and graphs for the loss functions. 

# The output prediction will of of .nii regardless of your original data type. I chose .nii because it is most commonly used and very easy to import/export later.
# There is a data augmentation function within this script and it will augment the data while importing so note that if you use it, you will literally double the memory required.
# All you need to do is add the dataset to the paths specified in the repo. change the names in this script and run it and you will be done.

from simple_unet_model_3d import simple_unet_model_3d   # That is the model script. Have a look at it too, although I already configured that so it should work without errors.
from keras.utils import normalize
import os
import sys
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from dataloader_mhd import load_itk
from datetime import datetime
from packaging import version
#import tensorflow as tf
from tensorflow import keras
import random
import csv
from sklearn.model_selection import train_test_split
import nibabel as nib

# Latex Font converter for matplotlib
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams.update({'font.size': 14})

UNIQUEID=datetime.now()
UNIQUEID=UNIQUEID.strftime("%m%d%H%M")
# import sys
# sys.exit(os.EX_OK)

# Configure these before Running the script

IMAGE_DIR = 'data/left_atrium/'
MASK_DIR = 'data/left_atrium/'
SIZE = 160                       # this is desired x/y square size, this script will automatically square crop the input images with respect to the center.
DEPTH = 16			 # this is the desired z length. Configure the function in the dataloader file to make sure the axes are correct for your samples.
DEPTH_MID = int(DEPTH/2)         # A slice will be taken along the z direction at the midpoint, to be displayed as 2D image in the graphs.
EPOCHS = 2                       # Number of epochs, my model converged at 600. 
TEST_SIZE = 0.2                  # The ratio of images from the total dataset that are used as validation or testing images
LOSS_FUNC = 'Jaccard'            # This is for naming the files and graphs, to change this, go to the simple unet 3d file
OPTIMIZER = 'SGD lr 0p01' 	 # This is for naming the files and graphs, to change this, go to the simple unet 3d file
FOLDER = '.\\Tests3d\\test_'+str(UNIQUEID)+'_EPO_'+str(EPOCHS)+'_Testsize_'+'0p2'+'_Loss_'+str(LOSS_FUNC)+'_Opt_'+str(OPTIMIZER)+'\\'
Model_Name = FOLDER+'model.hdf5'
#os.mkdir(os.path.abspath(os.getcwd())+'\\Tests',mode = 0o666)
os.mkdir(FOLDER,mode = 0o777)

print('directory created...')

#Function to augment data by shifting it in the x and y direction
def datashift(images,masks,shiftx,shifty):
	
	b = np.roll(images, shiftx, axis=1)
	c = np.roll(masks, shiftx, axis=1)
	
	if shiftx <0:
		b[:,shiftx:, :] = 0
		c[:,shiftx:, :] = 0
	else:
		b[:,:shiftx, :] = 0
		c[:,:shiftx, :] = 0

	b = np.roll(b, shifty, axis=0)
	c = np.roll(c, shifty, axis=0)
		
	if shifty <0:
		b[shifty:,:, :] = 0
		c[shifty:,:, :] = 0
	else:
		b[:shifty,:, :] = 0
		c[:shifty,:, :] = 0

	return(b,c)

## function to rotate the data

# def datarotate(degrees,image):
# 	theta = np.radians(degrees)
# 	v=gjmhjhj
# 	for n in range(DEPTH):
# 		r = np.array(( (np.cos(theta), -np.sin(theta)),
# 	               (np.sin(theta),  np.cos(theta)) ))
# 	# print('rotation matrix:')
# 	# print(r)
# 		v[:,:,n] = np.array(image[:,:,n])
# 		r.dot(v)

# Function to import the dataset into the memory, most of your initial problems will occur in this function as you will have different file names and paths.
# You will need to configure this file before your first run. Just debug this separately first and print the list dimensions to see if they are okay before running the whole script.

def importdata(IMAGE_DIR, MASK_DIR, SIZE):
    
    image_dataset = []  
    mask_dataset = []  

    images = os.listdir(IMAGE_DIR)
    for i, folder_name in enumerate(images):    
	
        if folder_name == 'data_readme.txt':
            continue
	
        path= IMAGE_DIR + folder_name
        
	image_name = '/image.mhd' # name of the input image
        
        if folder_name.startswith('a'): # names of the input mask/ground truth, there were two names hence I used an if. You will need to change this.
        	mask_name = '/gt_binary.mhd' 
        else:
        	mask_name = '/gt_std.mhd'

        image1, origin1, spacing1 = load_itk(path+image_name) # calling the function from the dataloader file to import the raw/mhd images. Change that function for diff data types.
	
        maxElement1 = np.amax(image1)
        image1= image1 * 255/maxElement1

        mask1, origin2, spacing2 = load_itk(path+mask_name) # calling the function from the dataloader file to import the raw/mhd masks/labe. Change that function for diff data types.

        image1=np.transpose(image1, (1, 2, 0))
        mask1=np.transpose(mask1, (1, 2, 0))

        #mask1 = mask1.astype(np.uint8)
        image1 = image1.astype(np.uint8)
        mask1 = (mask1 > 0)

        shape=image1.shape
        print(shape) 

        #new img dimentions after cropping
        dim1_start=int((shape[0]-SIZE)/2)
        dim1_end=int(shape[0]-dim1_start)
        dim2_start=int((shape[1]-SIZE)/2)
        dim2_end=int(shape[1]-dim2_start)
        dim3_start=int((shape[2]-DEPTH)/2)
        dim3_end=int(shape[2]-dim3_start-shape[2]%2)

        temp_image = image1[dim1_start:dim1_end,dim2_start:dim2_end,dim3_start:dim3_end]
        temp_mask = mask1[dim1_start:dim1_end,dim2_start:dim2_end,dim3_start:dim3_end]

        image_dataset.append(temp_image)
        mask_dataset.append(temp_mask)

        #print(len(image_dataset)) 
        print(temp_image.shape)    
    #(214214,320,320,1)
    #(214214,64,320,320,1)

    #image_dataset = np.array(image_dataset)
    #print(image_dataset.shape)

    #Normalize images
    image_dataset = np.expand_dims(normalize(image_dataset, axis=1),4)
    #D not normalize masks, just rescale to 0 to 1.
    mask_dataset = np.expand_dims((np.array(mask_dataset)),4) /1.

    return image_dataset, mask_dataset

image_dataset,mask_dataset = importdata(IMAGE_DIR, MASK_DIR, SIZE)


# THis was an experimental check, you don't need this.
def datacheck(X_train,y_train, FOLDER, DEPTH_X,image_number ):

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Test Image')
    plt.imshow(np.reshape(X_train[image_number,:,:,DEPTH_X], (SIZE, SIZE)), cmap='gray')
    plt.subplot(122)
    plt.title('Test Mask (Ground Truth)')
    plt.imshow(np.reshape(y_train[image_number,:,:,DEPTH_X], (SIZE, SIZE)), cmap='gray')
    plt.savefig(FOLDER+'\\'+str(DEPTH_X)+'\\'+'dataset_'+str(image_number)+'.png', dpi=300, bbox_inches='tight')
    plt.close()

DEPTH_X_list=[DEPTH_MID, DEPTH_MID+2] 

# THis was an experimental check, you don't need this. Comment this.
for DEPTH_X in DEPTH_X_list:
	os.mkdir(FOLDER+'\\'+str(DEPTH_X),mode = 0o777)
	for image_number in range(len(image_dataset)):
		datacheck(image_dataset, mask_dataset, FOLDER, DEPTH_X, image_number)

print('image_dataset shape:')
print(image_dataset.shape)

# Splitting the total data into data for training and data for testing randomly with a ratio that was set at the top.

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = TEST_SIZE, random_state = 0)

print('X_train shape:')
print(X_train.shape)

def plotsanity(X_train, FOLDER):

    image_number = random.randint(0, len(X_train)-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Test Image')
    plt.imshow(np.reshape(X_train[image_number,:,:,DEPTH_MID], (SIZE, SIZE)), cmap='gray')
    plt.subplot(122)
    plt.title('Test Mask (Ground Truth)')
    plt.imshow(np.reshape(y_train[image_number,:,:,DEPTH_MID], (SIZE, SIZE)), cmap='gray')
    plt.savefig(FOLDER+'plot1.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_model(image_dataset,EPOCHS,X_train,y_train,X_test,y_test,Model_Name,FOLDER):
    
    
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_DEPTH = image_dataset.shape[3]
    IMG_CHANNELS = image_dataset.shape[4]
    


    model = simple_unet_model_3d(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS)

    # with open(FOLDER+'Misc_'+str(UNIQUEID)+'.txt','w') as fh:
    # # Pass the file handle in as a lambda function to make it callable
    #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

    f = open(FOLDER+"epoch_time.txt", 'w')
    sys.stdout = f

    #If you want to load a previous model/weights. Use the line below. Note that this needs to be in the same folder.
	
    #model.load_weights('model.hdf5')    

    history_dice = model.fit(X_train, y_train,
                        batch_size = 32,
                        verbose=1,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        shuffle=False)

    model.save(Model_Name)

    #Evaluate the model
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy of UNet2d Model with Loss: " +str(LOSS_FUNC)+ " is = ", (acc * 100.0), "%")

    loss = history_dice.history['loss']
    val_loss = history_dice.history['val_loss']
    EPOCHS = range(1, len(loss) + 1)

    plt.plot(EPOCHS, loss, '#0080b3', label='Training loss') #green #80b300 pink #e60080 blue #0080b3
    plt.plot(EPOCHS, val_loss, '#e60080', label='Validation loss')
    plt.title('Training and validation Loss, Metric: '+str(LOSS_FUNC)+' ,Optimizer: ' + str(OPTIMIZER))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(FOLDER+'plot2.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

    jc = history_dice.history['dice_coef']
    #acc = history.history['accuracy']
    val_jc = history_dice.history['val_dice_coef']
    #val_acc = history.history['val_accuracy']

    plt.plot(EPOCHS, jc, '#0080b3', label='Training Accuracy')
    plt.plot(EPOCHS, val_jc, '#e60080', label='Validation Accuracy')
    plt.title('Training and validation Accuracy, Metric: '+str(LOSS_FUNC)+' ,Optimizer: ' + str(OPTIMIZER))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Metric')
    plt.legend()
    plt.savefig(FOLDER+'plot3.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

    #IOU
    y_pred=model.predict(X_test)
    y_pred_thresholded = y_pred > 0.5

    intersection = np.logical_and(y_test, y_pred_thresholded)
    union = np.logical_or(y_test, y_pred_thresholded)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)

    #######################################################################
    #Predict on a few images

    test_img_number = random.randint(0, len(X_test)-1)
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_norm=test_img[:,:,:,0][:,:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0,:,:,:,0] > 0.5).astype(np.uint8)

    path='.\\data\\evaluate\\image.mhd'
    test_img_other, origin3, spacing3 = load_itk(path)
    print('Spacing of test image: ', spacing3)
    test_img_other=np.transpose(test_img_other, (1, 2, 0))

    shape=test_img_other.shape
    dim1_start=int((shape[0]-SIZE)/2)
    dim1_end=int(shape[0]-dim1_start)
    dim2_start=int((shape[1]-SIZE)/2)
    dim2_end=int(shape[1]-dim2_start)
    dim3_start=int((shape[2]-DEPTH)/2)
    dim3_end=int(shape[2]-dim3_start-shape[2]%2)

    test_img_other = test_img_other[dim1_start:dim1_end,dim2_start:dim2_end,dim3_start:dim3_end]

    #test_img_other = image_dataset[0,:,:,:,:]
    # test_img_other = cv2.imread('data/test_images/01-1_256.tif', 0)
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),3)
    test_img_other_norm=test_img_other_norm[:,:,:,0][:,:,:,None]
    test_img_other_input=np.expand_dims(test_img_other_norm, 0)

    #Predict and threshold for values above 0.5 probability
    prediction_other = (model.predict(test_img_other_input)[0,:,:,:,0] > 0.5).astype(np.uint8)


    plt.figure(figsize=(16, 8))
    plt.title('Model Result with loss: '+str(LOSS_FUNC)+' and optimizer: '+str(OPTIMIZER))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,DEPTH_MID,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,DEPTH_MID,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction[:,:,DEPTH_MID], cmap='gray')
    plt.subplot(234)
    plt.title('External Image')
    plt.imshow(test_img_other[:,:,DEPTH_MID], cmap='gray')
    plt.subplot(235)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other[:,:,DEPTH_MID], cmap='gray')
    plt.savefig(FOLDER+'plot4.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

    with open(FOLDER+'summary.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch'])
        writer.writerow(EPOCHS)
        writer.writerow(['Loss'])
        writer.writerow(loss)
        writer.writerow(['Validation_Loss'])
        writer.writerow(val_loss)
        writer.writerow(['Accuracy_Metric'])
        writer.writerow(jc)
        writer.writerow(['Validation_Accuracy'])
        writer.writerow(val_jc)
        writer.writerow(['Accuracy_Percentage'])
        writer.writerow([acc*100.0])
        writer.writerow(['Iou_score'])
        writer.writerow([iou_score])
        writer.writerow(['Loss_Function'])
        writer.writerow([LOSS_FUNC])
        writer.writerow(['Optimizer'])
        writer.writerow([OPTIMIZER])
    
    plt.imsave('output.png', prediction_other[:,:,DEPTH_MID], cmap='gray')
    

    nifty_img = nib.Nifti1Image(prediction_other, affine=np.eye(4))
    nib.save(nifty_img, FOLDER+'prediction.nii') 

    f.close()   
