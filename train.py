

from simple_unet_model_with_jacard import simple_unet_model_with_jacard   #Use normal unet model
from keras.utils import normalize
import os
import sys
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from dataloader_mhd import load_itk
from arraytesting import loaddataB
from datetime import datetime
from packaging import version
#import tensorflow as tf
from tensorflow import keras
import random

# Latex Font converter for matplotlib
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams.update({'font.size': 14})

UNIQUEID=datetime.now()
UNIQUEID=UNIQUEID.strftime("%m%d%H%M")
# import sys
# sys.exit(os.EX_OK)
IMAGE_DIR = 'data/left_atrium/'
MASK_DIR = 'data/left_atrium/'
SIZE = 160
EPOCHS = 2
TEST_SIZE = 0.9
LOSS_FUNC = 'Jaccard'
OPTIMIZER = 'Adam'
FOLDER = '.\\Tests\\Test_'+str(UNIQUEID)+'_EPO_'+str(EPOCHS)+'_Testsize_'+str(TEST_SIZE)+'_Loss_'+str(LOSS_FUNC)+'_Opt_'+str(OPTIMIZER)+'\\'
Model_Name = FOLDER+'Model_'+str(UNIQUEID)+'.hdf5'
#os.mkdir(os.path.abspath(os.getcwd())+'\\Tests',mode = 0o666)
os.mkdir(FOLDER,mode = 0o666)

print('directory created...')
def importdata(IMAGE_DIR, MASK_DIR, SIZE):
    
    image_dataset = []  #Here, we are using a list format.
    mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    images = os.listdir(IMAGE_DIR)
    for i, folder_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
        
        if folder_name == 'data_readme.txt':
            continue
        path= IMAGE_DIR + folder_name
        image_name = '/image.mhd'
        
        if folder_name.startswith('a'):
        	mask_name = '/gt_binary.mhd'
        else:
        	continue#mask_name = '/gt_std.mhd'

        image1, origin1, spacing1 = load_itk(path+image_name)
        maxElement1 = np.amax(image1)
        image1= image1 * 255/maxElement1

        mask1, origin2, spacing2 = load_itk(path+mask_name)

        #mask1 = mask1.astype(np.uint8)
        image1 = image1.astype(np.uint8)
        mask1 = (mask1 > 0)

        shape=image1.shape

        for i in range(shape[0]):

            #new img dimentions after cropping
            crop_dim1_start=int((shape[1]-SIZE)/2)
            crop_dim1_end=int(shape[1]-crop_dim1_start)
            crop_dim2_start=int((shape[2]-SIZE)/2)
            cropy_dim2_end=int(shape[1]-crop_dim2_start)

            temp_image = image1[i,crop_dim1_start:crop_dim1_end,crop_dim2_start:cropy_dim2_end]
            temp_mask = mask1[i,crop_dim1_start:crop_dim1_end,crop_dim2_start:cropy_dim2_end]

            maxElement4 = np.amax(temp_mask)
            if maxElement4 ==0:
            	continue

            image_dataset.append(np.array(temp_image))
            mask_dataset.append(np.array(temp_mask))
    
    #Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
    #D not normalize masks, just rescale to 0 to 1.
    mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

    return image_dataset, mask_dataset

image_dataset,mask_dataset = importdata(IMAGE_DIR, MASK_DIR, SIZE)

print('image_dataset shape:')
print(image_dataset.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = TEST_SIZE, random_state = 0)

print('X_train shape:')
print(X_train.shape)

def plotsanity(X_train, FOLDER):

    image_number = random.randint(0, len(X_train)-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Test Image')
    plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE)), cmap='gray')
    plt.subplot(122)
    plt.title('Test Mask (Ground Truth)')
    plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE)), cmap='gray')
    plt.savefig(FOLDER+'plot1.png', dpi=300, bbox_inches='tight')
    plt.close()

#plot a sanity check to see input array
plotsanity(X_train, FOLDER)

def get_model(image_dataset,EPOCHS,X_train,y_train,X_test,y_test,Model_Name,FOLDER):
    
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH  = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]
    
    model = simple_unet_model_with_jacard(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # with open(FOLDER+'Misc_'+str(UNIQUEID)+'.txt','w') as fh:
    # # Pass the file handle in as a lambda function to make it callable
    #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

    f = open(FOLDER+"Epoch_time_"+UNIQUEID+".txt", 'w')
    sys.stdout = f
    #If starting with pre-trained weights.
    #model.load_weights('mitochondria_gpu_tf1.4.hdf5')

    history_jacard = model.fit(X_train, y_train,
                        batch_size = 16,
                        verbose=1,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test),
                        shuffle=False)

    model.save(Model_Name)

    #Evaluate the model
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy of UNet2d Model with Loss: " +str(LOSS_FUNC)+ " is = ", (acc * 100.0), "%")

    loss = history_jacard.history['loss']
    val_loss = history_jacard.history['val_loss']
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

    jc = history_jacard.history['jacard_coef']
    #acc = history.history['accuracy']
    val_jc = history_jacard.history['val_jacard_coef']
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
    #model = get_model()
    #model.load_weights('mitochondria_with_jacard_50_plus_50_EPOCHS.hdf5') #Trained for 50 EPOCHS and then additional 100
    #model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 EPOCHS

    test_img_number = random.randint(0, len(X_test-1))
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

    test_img_other = image_dataset[40,:,:,:]
    # test_img_other = cv2.imread('data/test_images/01-1_256.tif', 0)
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1),2)
    test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
    test_img_other_input=np.expand_dims(test_img_other_norm, 0)

    #Predict and threshold for values above 0.5 probability
    prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.5).astype(np.uint8)


    plt.figure(figsize=(16, 8))
    plt.title('Model Result with loss: '+str(LOSS_FUNC)+' and optimizer: '+str(OPTIMIZER))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.subplot(234)
    plt.title('External Image')
    plt.imshow(test_img_other, cmap='gray')
    plt.subplot(235)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other, cmap='gray')
    plt.savefig(FOLDER+'plot4.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

    import csv
    with open(FOLDER+'summary_'+str(UNIQUEID)+'.csv', 'w', newline='') as file:
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
    
    f.close()   
get_model(image_dataset,EPOCHS,X_train,y_train,X_test,y_test,Model_Name,FOLDER)

