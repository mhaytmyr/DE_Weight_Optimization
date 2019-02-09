
import keras as K
from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, AveragePooling2D
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Add, Conv2D
from keras.layers import Lambda, Input, BatchNormalization, Concatenate
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications import ResNet50, VGG19
from config import *
import numpy as np


def conv_net():
    input_img = Input(shape=(H,W,C));
    conv1 = Conv2D(filter1,(3,3),padding='same',use_bias=False)(input_img)
    conv1 = Activation('relu')(conv1);
    conv2 = Conv2D(filter2,(2,2),padding='same',use_bias=False)(conv1)
    conv2 = Activation('relu')(conv2);
    #conv3 = Conv2D(filter3,(1,1),padding='same',use_bias=False)(conv2)
    #conv3 = Activation('linear')(conv3);
    #conv3 = Concatenate(axis=3)([input_img, conv2])

    out = Conv2D(1,(1,1),padding='same',use_bias=False)(conv2);
    out = Activation('relu')(out);

    model = Model(input_img,out);

    return model


def fcn_net():
    input_img = Input(shape=(H,W,C));
    conv1 = Conv2D(filter1,(downKernel,downKernel),padding='same',use_bias=False,kernel_initializer='glorot_uniform')(input_img)
    conv1 = BatchNormalization(epsilon=1e-5)(conv1)    
    #conv1 = Activation('tanh')(conv1);
    conv1 = Activation('linear')(conv1);
    conv2 = Conv2D(filter2,(downKernel,downKernel),padding='same',use_bias=False,kernel_initializer='glorot_uniform')(conv1)
    conv2 = BatchNormalization(epsilon=1e-5)(conv2)    
    #conv2 = Activation('tanh')(conv2);	
    conv2 = Activation('linear')(conv2);
    
    out = Conv2D(1,(downKernel,downKernel),padding='same',use_bias=False,kernel_initializer='glorot_uniform')(conv2);
    #out = BatchNormalization(epsilon=1e-5)(out)    
    out = Activation('linear')(out);

    model = Model(input_img,out);

    return model


def down(filters, downKernel, input_):
    down_ = Conv2D(filters, (downKernel, downKernel), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4,axis=-1)(down_)
    down_ = Activation('relu')(down_)
    down_ = Conv2D(filters, (downKernel, downKernel), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4,axis=-1)(down_)
    down_res = Activation('relu')(down_)
    #down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_res)

    return down_res, input_

def resUp(filters, upKernel, input_, down_, pad='same',padWidth=False):
    if padWidth:
        input_ = ZeroPadding2D(padWidth,data_format='channels_last')(input_)
    #up_ = UpSampling2D((2, 2))(input_)

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = Activation('relu')(up_)
    up_ = Concatenate(axis=-1)([down_, up_])
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = Activation('relu')(up_)
    return up_

def unet_1024():
    x = Input(shape=(H,W,C))

    down0a, down0_res = down(filter1, downKernel, x)
    print('Down 0 ',down0a.shape)
    down1a, down1_res = down(filter2, downKernel, down0a)
    print('Down 1 ',down1a.shape)
    down2a, down2_res = down(filter3, downKernel, down1a)

    #add two layers
    #center = Conv2D(1024, (1, 1), padding='same')(down4a)
    #center = BatchNormalization(epsilon=1e-4)(center)
    #center = Activation('relu')(center)
    #center = Conv2D(1024, (3, 3), padding='same')(center)
    #center = Dropout(0.5)(center)
    #center = BatchNormalization(epsilon=1e-4)(center)
    #center = Activation('relu')(center)

    up2 = resUp(filter3, upKernel, down2a, down2_res)
    print('Up 2 ',up2.shape)
    up1 = resUp(filter1, upKernel, up2, down1_res)
    print('Up 1 ',up1.shape)
    up0 = resUp(filter1, upKernel, up1, down0_res)
    print('Up 0 ',up0.shape)

    y = Conv2D(1,(upKernel,upKernel),activation='linear',padding='same')(up0)

    #create model
    model = Model(x,y)

    return model

def upsilon_net():

    #get input layer
    x = Input(shape=(H,W,C));

    #split input layer into two halves
    x1 = Lambda(lambda x: x[:,:,:,0][:,:,:,np.newaxis])(x);
    x2 = Lambda(lambda x: x[:,:,:,1][:,:,:,np.newaxis])(x);

    #now convolve each layer separately and concatenate them on latent
    #first input
    conv1a = Conv2D(filter1, (downKernel, downKernel), activation='linear', padding='same', input_shape=(H,W,1))(x1)
    conv2a = Conv2D(filter1, (downKernel+2, downKernel+2), activation='linear', padding='same',input_shape=(H,W,1))(x1)
    conv3a = Conv2D(filter1, (downKernel+4, downKernel+4), activation='linear', padding='same',input_shape=(H,W,1))(x1)

    #concatenate first input
    concat1 = K.layers.Concatenate(axis=3)([conv1a,conv2a,conv3a,x1]);

    conv1b = Conv2D(filter1, (downKernel, downKernel), activation='linear', padding='same',input_shape=(H,W,1))(x2)
    conv2b = Conv2D(filter1, (downKernel+2, downKernel+2), activation='linear', padding='same',input_shape=(H,W,1))(x2)
    conv3b = Conv2D(filter1, (downKernel+4, downKernel+4), activation='linear', padding='same',input_shape=(H,W,1))(x2)

    #concatenate first input
    concat2 = K.layers.Concatenate(axis=3)([conv1b,conv2b,conv3b,x2]);

    #decoder
    #concatenate layers
    #concat = K.layers.add([concat1, concat2])

    y = Conv2D(1,(upKernel,upKernel),activation='linear',padding='same')(concat)

    #create model
    model = Model(x,y)

    return model

def seg_net():
    input_img = Input(shape=(H,W,C));

    #encoder 1
    down1 = Conv2D(filter1, (downKernel, downKernel), padding='same')(input_img)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    #down1 = Conv2D(filter1, (downKernel, downKernel), padding='same')(down1)
    #down1 = BatchNormalization()(down1)
    #down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((downPool, downPool), strides=(downPool, downPool))(down1)
    #print("Conv1 ", down1_pool.shape)

    #encoder 2
    down2 = Conv2D(filter2, (downKernel, downKernel), padding='same')(down1_pool)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    #down2 = Conv2D(filter2, (downKernel, downKernel), padding='same')(down2)
    #down2 = BatchNormalization()(down2)
    #down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((downPool, downPool), strides=(downPool, downPool))(down2)
    #encoder 3
    down3 = Conv2D(filter3, (downKernel, downKernel), padding='same')(down2_pool)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    #down3 = Conv2D(filter3, (downKernel, downKernel), padding='same')(down3)
    #down3 = BatchNormalization()(down3)
    #down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((downPool, downPool), strides=(downPool, downPool))(down3)
    #center layer
    center = Conv2D(1024, (3, 3), padding='same')(down3_pool)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    #center = Conv2D(1024, (3, 3), padding='same')(center)
    #center = BatchNormalization()(center)
    #center = Activation('relu')(center)
    #decoder 3
    up3 = UpSampling2D((upPool, upPool))(center)
    up3 = Concatenate(axis=3)([down3, up3]) # <-- Use output from corresponding encoder layer
    up3 = Conv2D(filter3, (upKernel, upKernel), padding='same')(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    #up3 = Conv2D(filter3, (upKernel, upKernel), padding='same')(up3)
    #up3 = BatchNormalization()(up3)
    #up3 = Activation('relu')(up3)
    #up3 = Conv2D(filter3, (upKernel, upKernel), padding='same')(up3)
    #up3 = BatchNormalization()(up3)
    #up3 = Activation('relu')(up3)
    #decoder 2
    up2 = UpSampling2D((upPool, upPool))(up3)
    up2 = Concatenate(axis=3)([down2, up2]) # <-- Use output from corresponding encoder layer
    up2 = Conv2D(filter2, (upKernel, upKernel), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    #up2 = Conv2D(filter2, (upKernel, upKernel), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    #up2 = Activation('relu')(up2)
    #up2 = Conv2D(filter2, (upKernel, upKernel), padding='same')(up2)
    #up2 = BatchNormalization()(up2)
    #up2 = Activation('relu')(up2)
    #decoder 1
    up1 = UpSampling2D((upPool, upPool))(up2)
    up1 = Concatenate(axis=3)([down1, up1]) # <-- Use output from corresponding encoder layer
    up1 = Conv2D(filter1, (upKernel, upKernel), padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    #up1 = Conv2D(filter1, (upKernel, upKernel), padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    #up1 = Activation('relu')(up1)
    #up1 = Conv2D(filter1, (upKernel, upKernel), padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    #up1 = Activation('relu')(up1)
    #final layer
    out = Conv2D(1,(upKernel,upKernel),padding='same',activation='linear')(up1)

    #create model
    model = Model(input_img,out);
    print(model.summary())

    return model


def patch_encoder_decoder():

    #create sequential model
    model = Sequential()

    ##encoder
    model.add(Conv2D(filter1,(downKernel,downKernel),padding='same',activation='linear',input_shape=(H,W,C)));
    #model.add(LeakyReLU(alpha=0.03));
    #model.add(MaxPooling2D(pool_size=(downPool,downPool), padding='same'));
    model.add(Conv2D(filter2,(downKernel,downKernel),activation='linear',padding='same'));
    #model.add(LeakyReLU(alpha=0.03));
    #model.add(MaxPooling2D(pool_size=(downPool,downPool), padding='same'));
    #model.add(Conv2D(filter3,(downKernel,downKernel),activation='relu', padding='same',
    #    strides=(downPool,downPool)));
    #model.add(MaxPooling2D(pool_size=(downPool,downPool), padding='same'));
    ##decoder
    #model.add(Conv2D(filter3, (upKernel,upKernel), activation='relu', padding='same'))
    #model.add(UpSampling2D((upPool,upPool)))
    #model.add(Conv2D(filter2, (upKernel,upKernel), activation='relu', padding='same'))
    #model.add(UpSampling2D((upPool,upPool)))
    #model.add(Conv2D(filter1, (upKernel,upKernel), activation='relu', padding='same'))
    #model.add(UpSampling2D((upPool,upPool)))
    #add final layer with single channel output
    model.add(Conv2D(1, (upKernel,upKernel),activation='linear', padding='same'))

    return model

def get_encoder_cbct(input_img,down_kernel,down_pool):

    x = Conv2D(layer1, (down_kernel, down_kernel), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((down_pool, down_pool), padding='same')(x)
    conv1_shape = x.shape;
    print("Conv1 ", conv1_shape)

    x = Conv2D(layer2, (down_kernel, down_kernel), activation='elu', padding='same')(x)
    x = MaxPooling2D((down_pool, down_pool), padding='same')(x)
    conv2_shape = x.shape;
    print("Conv2 ", conv2_shape)

    x = Conv2D(layer3, (down_kernel, down_kernel), activation='elu', padding='same')(x)
    x = MaxPooling2D((down_pool, down_pool), padding='same')(x)
    conv3_shape = x.shape;
    print("Conv3 ", conv3_shape)


    ##### MODEL 1: ENCODER #####
    encoder = Model(input_img, x)

    return encoder, x, int(x.shape[1]), int(x.shape[2]), layer3

def get_decoder_cbct(conv_shape1, conv_shape2, layer, up_kernel, up_pool, H,W):

    input_z = Input(shape=(conv_shape1, conv_shape2, layer))
    #input_z = Input(shape=(latent_dim,))
    #first convert image to original shape
    #x = Dense(conv_shape1*conv_shape2*layer, activation='elu')(input_z)
    #x = Reshape((conv_shape1,conv_shape2,layer))(x)


    x = Conv2D(layer3, (up_kernel,up_kernel), activation='elu', padding='same')(input_z)
    x = UpSampling2D((up_pool,up_pool))(x)
    print("De Conv3 ", x.shape)

    x = Conv2D(layer2, (up_kernel,up_kernel), activation='elu', padding='same')(x)
    x = UpSampling2D((up_pool,up_pool))(x)
    print("De Conv2 ", x.shape)

    x = Conv2D(layer1, (up_kernel,up_kernel), activation='relu', padding='same')(x)
    x = UpSampling2D((up_pool,up_pool))(x)
    print("De Conv3 ", x.shape)

    decoded = Conv2D(1, (up_kernel,up_kernel), activation='relu', padding='same')(x)
    print("Output layer", decoded.shape)

    ##### MODEL 2: DECODER #####
    decoder = Model(input_z, decoded)

    return decoder, x
