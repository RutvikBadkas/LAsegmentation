# Modified U-net CNN model 
# import all the packages.
from keras import optimizers
from keras.models import Model
from keras import Input
from keras.layers import Dot, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K

# Dice Coefficient Function as a measure of accuracy.

def dice_coef(y_true, y_pred):
    smooth=0.0000001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Dice Coefficient Function as Loss function.

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Jaccard Index Function

def jacard_coef(y_true, y_pred):
    # y_true = y_true[0, :, :, :, :]
    # y_pred  = y_pred[0, :, :, :, :]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# Jaccard Index Function as Loss function.

def jacard_coef_loss(y_true, y_pred):
    return 1-jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function

# I didnt derive Jaccard or dice from each other because I didnt want to merge the uncertainties of one into that of another as I compare both the metrics in my research.
# Below is the main model funciton. This is called in the training file. Changes in the architecture can be very easily made by just removing layers separated by an empty line.
# Note: be careful because chances are, you will mismatch the dimensions from one layer to another when you edit the model. So do the calculations first of the stride size and 
# convolutional kernal sizes to see if the output from the previous layer matches the input into this layer. See my research paper for a proposed model with less layers, use
# that as a basis to reduce the number of layers.
# Also: do NOT add more layers, the model is very suceptible to overfitting. Would recommend experimenting with the dataset first. I have added training scripts with aug functions
# to this repo so try that first. 

def simple_unet_model_3d(IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    
    K.set_image_data_format('channels_last')
    inputs = Input((IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
    
    # This is the 1x1x1 convolution layer that I mentioned in my paper, sigmoid only works because the situation is binary, softMax will be required if you add multiple classes
        
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # I recommend you start with the the Adam optimiser first. I only used SGD because it was very case specific and I needed to fine tune the optimizer parameters. 
    # if you get a lot of spikes in the model, and you don't want to reduce the learning rate, then change the clipnorm paramter in line 116.
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
    #optimizer = optimizers.Adam(learning_rate=1)
       
    optimizer = optimizers.SGD(lr= 0.01, momentum=0.9, nesterov=True, clipnorm=1.)
    
    model.compile(optimizer = optimizer, loss = [dice_coef_loss], metrics = [dice_coef])

    model.summary()
    
    return model
