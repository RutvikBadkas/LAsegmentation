
from tensorflow import keras

from unet3D_model_with_jacard import simple_unet_model_with_jacard   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from dataloader_mhd import load_itk
from arraytesting import loaddataB
from datetime import datetime
from packaging import version
#import tensorflow as tf

image_directory = 'data/left_atrium/'
mask_directory = 'data/left_atrium/'

# Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# --------------------------------------------------------------------------------------------------------change
# --------------------------------------------------------------------------------------------------------change
SIZE = 64
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

# 3d image in 32000x320 (2d format) - single patient
# list of this 3d image in 2d format
# 320x320 as a single element in the list
# so there will be two for loops, one for different patients, and one for the 100 slices for each patient

# 2d images in 256 x 256
# list of these 2d miages

images = os.listdir(image_directory)
for i, folder_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if folder_name == 'data_readme.txt':
        continue
    path= image_directory + folder_name
    image_name = '/image.mhd'
    
    if folder_name.startswith('a'):
        mask_name = '/gt_binary.mhd'
    else:
        continue#mask_name = '/gt_std.mhd'

    image1, origin1, spacing1 = load_itk(path+image_name)
    maxElement1 = np.amax(image1)
    image1= image1 * 255/maxElement1
    maxElement3 = np.amax(image1)
    print(maxElement1)
    print(maxElement3)

    print('image:' ,image1.shape) 
    mask1, origin2, spacing2 = load_itk(path+mask_name)

    shape=image1.shape
       
    print('mask:' ,mask1.shape)    
    
    mask1 = mask1.astype(np.uint8)
    image1 = image1.astype(np.uint8)
    mask1 = (mask1 > 0)
    # maxElement2 = np.amax(mask1)
    # mask1 * 255/maxElement2
    # print(maxElement2)
    # #print(image_directory+image_name)
    # image = cv2.imread(image_directory+image_name, 0)
    # image = Image.fromarray(image)
    # image = image.resize((SIZE, SIZE))

    #for i in range(shape[0]):
        
        
    temp_image = image1[30:62,32:288,32:288]
    temp_mask = mask1[30:62,32:288,32:288]

        # maxElement4 = np.amax(temp_mask)
        # if maxElement4 ==0:
        #     continue
    temp_image=np.array(temp_image)
    print(temp_image.shape)
    temp_mask=np.array(temp_mask)
    print(temp_mask.shape)

    image_dataset.append(np.array(temp_image))
    mask_dataset.append(np.array(temp_mask))
        # print(len(mask_dataset))
# image_dataset = np.array(image_dataset)
# print(image_dataset.shape)

# --------------------------------------------------------------------------------------------------------change
# images = os.listdir(image_directory)
# for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
#     if (image_name.split('.')[1] == 'tif'):
#         #print(image_directory+image_name)
#         image = cv2.imread(image_directory+image_name, 0)
#         image = Image.fromarray(image)
#         image = image.resize((SIZE, SIZE))
#         image_dataset.append(np.array(image))



# image_dataset, origin1, spacing1 = load_itk('image.mhd')
# maxElement1 = np.amax(image_dataset)
# image_dataset * 255/maxElement1
# # --------------------------------------------------------------------------------------------------------change


# #Iterate through all images in Uninfected folder, resize to 64 x 64
# #Then save into the same numpy array 'dataset' but with label 1

# # --------------------------------------------------------------------------------------------------------change

# # masks = os.listdir(mask_directory)
# # for i, image_name in enumerate(masks):
# #     if (image_name.split('.')[1] == 'tif'):
# #         image = cv2.imread(mask_directory+image_name, 0)
# #         image = Image.fromarray(image)
# #         image = image.resize((SIZE, SIZE))
# #         mask_dataset.append(np.array(image))


# mask_dataset, origin2, spacing2 = load_itk('gt_binary.mhd')
# maxElement2 = np.amax(mask_dataset)
# mask_dataset * 255/maxElement2

# --------------------------------------------------------------------------------------------------------change
#Normalize images
image_dataset=np.array(image_dataset)
print('Img shape:')
print(image_dataset.shape)

image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),4)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),4) /1.
print(image_dataset.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

print('X_train shape:')
print(X_train.dtype)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
# --------------------------------------------------------------------------------------------------------change
plt.imshow(np.reshape(X_train[image_number,30], (SIZE, SIZE)), cmap='gray')
plt.subplot(122)

# --------------------------------------------------------------------------------------------------------change
plt.imshow(np.reshape(y_train[image_number,30], (SIZE, SIZE)), cmap='gray')
plt.show()

# import sys
# sys.exit(os.EX_OK)
###############################################################
IMG_DEPTH = image_dataset.shape[1]
IMG_HEIGHT = image_dataset.shape[2]
IMG_WIDTH  = image_dataset.shape[3]
IMG_CHANNELS = image_dataset.shape[4]

def get_model():
    return simple_unet_model_with_jacard(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


#If starting with pre-trained weights.
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')

history_jacard = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=5,
                    validation_data=(X_test, y_test),
                    shuffle=False)

# # https://youtu.be/BNPW1mYbgS4

# """
# Training and testing for semantic segmentation (Unet) of mitochondria
# Uses standard Unet framework with Jacard for loss. 
# Dataset info: Electron microscopy (EM) dataset from
# https://www.epfl.ch/labs/cvlab/data/data-em/
# Patches of 256x256 from images and labels 
# have been extracted (via separate program) and saved to disk. 
# This code uses 256x256 images/masks.
# """

# from simple_unet_model_with_jacard import simple_unet_model_with_jacard   #Use normal unet model
# from simple_unet_model import simple_unet_model   #Use normal unet model
# from arraytesting import loaddataB
# from keras.utils import normalize
# import os
# import cv2
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt


# image_directory = 'data/left_atrium/'
# mask_directory = 'data/left_atrium/'

# SIZE = 160
# image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
# mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

# # images = os.listdir(image_directory)
# # for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
# #     if (image_name.split('.')[1] == 'tif'):
# #         #print(image_directory+image_name)
# #         image = cv2.imread(image_directory+image_name, 0)
# #         image = Image.fromarray(image)
# #         image = image.resize((SIZE, SIZE))
# #         image_dataset.append(np.array(image))

# # #Iterate through all images in Uninfected folder, resize to 64 x 64
# # #Then save into the same numpy array 'dataset' but with label 1

# # masks = os.listdir(mask_directory)
# # for i, image_name in enumerate(masks):
# #     if (image_name.split('.')[1] == 'tif'):
# #         image = cv2.imread(mask_directory+image_name, 0)
# #         image = Image.fromarray(image)
# #         image = image.resize((SIZE, SIZE))
# #         mask_dataset.append(np.array(image))
# image_dataset,mask_dataset = loaddataB('Loading!')

# #Normalize images
# image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
# #D not normalize masks, just rescale to 0 to 1.
# mask_dataset = np.expand_dims((np.array(mask_dataset)),3)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

# X_train_quick_test, X_test_quick_test, y_train_quick_test, y_test_quick_test = train_test_split(X_train, y_train, test_size = 0.9, random_state = 0)

# #Sanity check, view few mages
# import random
# import numpy as np
# image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(X_train[image_number], (160, 160)), cmap='gray')
# plt.subplot(122)
# plt.imshow(np.reshape(y_train[image_number], (160, 160)), cmap='gray')
# plt.show()

# ###############################################################
# IMG_HEIGHT = image_dataset.shape[1]
# IMG_WIDTH  = image_dataset.shape[2]
# IMG_CHANNELS = image_dataset.shape[3]

# def get_jacard_model():
#     return simple_unet_model_with_jacard(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# model_jacard = get_jacard_model()

# # def get_standard_model():
# #     return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# # model_standard = get_standard_model()

# #If starting with pre-trained weights. 
# #model.load_weights('mitochondria_with_jacard_50_epochs.hdf5')
# print(model_jacard)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)


# history_jacard = model_jacard.fit(X_train, y_train, 
#                     batch_size = 16, 
#                     verbose=1, 
#                     epochs=10, 
#                     validation_data=(X_test, y_test), 
#                     shuffle=False)

# history_standard = model_standard.fit(X_train_quick_test, y_train_quick_test, 
#                     batch_size = 16, 
#                     verbose=1, 
#                     epochs=10, 
#                     validation_data=(X_test, y_test), 
#                     shuffle=False)

model.save('test_3d_epoch_10_data_30.hdf5')
# model_standard.save('test_standard_epoch_10_data_30.hdf5')
#To be compared with trained model mitochondria_gpu_tf1.4.hdf5
# that just used cross entropy for loss and accuracy for metric. 

############################################################
#Evaluate the model


	# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy of Jacard Model is = ", (acc * 100.0), "%")

# 	# evaluate model
# _, acc = model_standard.evaluate(X_test, y_test)
# print("Accuracy of Standard Model is = ", (acc * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history_jacard.history['loss']
val_loss = history_jacard.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

jc = history_jacard.history['jacard_coef']
#acc = history.history['accuracy']
val_jc = history_jacard.history['val_jacard_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, jc, 'y', label='Training Jacard Coeff.')
plt.plot(epochs, val_jc, 'r', label='Validation Jacard Coeff.')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jacard Coefficient')
plt.legend()
plt.show()


# ###
# #plot the training and validation accuracy and loss at each epoch
# loss = history_standard.history['loss']
# val_loss = history_standard.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history_standard.history['acc']
# #acc = history.history['accuracy']
# val_acc = history_standard.history['val_acc']
# #val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
# plt.title('Training and validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

##################################

# model = model_jacard  #Assign model to one of the models instead of changing the entire code for testing. 
#model = model_standard  #Assign model to one of the models instead of changing the entire code for testing. 

#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('mitochondria_with_jacard_50_plus_50_epochs.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number = random.randint(0, len(X_test-1))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:,0][:,:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,:,0] > 0.5).astype(np.uint8)

test_img_other = image_dataset[1,:,:,:,:]
# test_img_other = cv2.imread('data/test_images/01-1_256.tif', 0)
test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),3)
test_img_other_norm=test_img_other_norm[:,:,:,0][:,:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
prediction_other = (model.predict(test_img_other_input)[0,:,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[30,:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[30,:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()

#plt.imsave('input.jpg', test_img[:,:,0], cmap='gray')
#plt.imsave('output2.jpg', prediction_other, cmap='gray')