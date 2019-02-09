
import keras
import tensorflow as tf
import numpy as np
import glob

from helper_to_models import *
from helper_to_prepare_data import *
from config import *
from keras.models import load_model
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error

def SSIM_cs(y_true, y_pred):
    patches_true = tf.extract_image_patches(y_true, ksizes=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                                        rates=[1, 1, 1, 1], padding="VALID")
    patches_pred = tf.extract_image_patches(y_pred, ksizes=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                                        rates=[1, 1, 1, 1], padding="VALID")

    var_true = K.var(patches_true, axis=(1,2))
    var_pred = K.var(patches_pred, axis=(1,2))
    mean_true = K.mean(patches_true, axis=(1,2))
    mean_pred = K.mean(patches_pred, axis=(1,2))
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    covar_true_pred = K.mean(patches_true*patches_pred,axis=(1,2))-mean_true*mean_pred;

    c1 = (0.01*11.09) ** 2
    c2 = (0.03*11.09) ** 2
    #contrast = (2 * std_pred * std_true + c2)/(var_pred+var_true+c2);
    #lumi = (2 * mean_pred * mean_true + c1)/(mean_pred**2+mean_true**2+c1);
    #struct = (covar_true_pred+c2/2)/(std_true*std_pred+c2/2);

    ssim = (2 * mean_true*mean_pred + c1) * (2*covar_true_pred+c2)
    denom = (mean_pred**2+mean_true**2+c1) * (var_pred + var_true + c2)
    ssim /= denom

    #ssim = contrast*struct;
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return K.mean(ssim)

def mix_loss(y_true,y_pred):
    alpha = 0.3;
    #return alpha*mean_squared_error(y_true,y_pred)+(1-alpha)*(1-SSIM_cs(y_true,y_pred))
    beta,gamma = 1,3
    return beta*mean_absolute_error(y_true,y_pred) #+gamma*(1-SSIM_cs(y_true,y_pred))


def train_patch_cbct(trainGen,valGen):

    try:
        model = load_model('./checkpoint/'+modelName+".h5",custom_objects={'mix_loss':mix_loss,'SSIM_cs':SSIM_cs});
        print("Laoading model...");

    except Exception as e:
        print(e);
        print("Creating new model...")
        #model = conv_net();
        model = fcn_net();

    print(model.summary())
    #define optimizer
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.95,decay=8)
    #sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=False)

    rmsprop = keras.optimizers.RMSprop(
            lr=0.0001, #global learning rate,
            rho=0.95, #exponential moving average; r = rho*initial_accumilation+(1-rho)*current_gradient
            epsilon=1e-6, #small constan to stabilize division by zero
            #decay = lrDecayRate
            )

    #compile model
    model.compile(optimizer=rmsprop,
            loss = [mix_loss],metrics=[SSIM_cs]
            );

    #define callbacks
    checkpoint = keras.callbacks.ModelCheckpoint("./checkpoint/"+modelName+".h5",
            monitor='mix_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    #fit model and store history
    hist = model.fit_generator(trainGen,
            steps_per_epoch = stepsPerEpoch,
            epochs=numEpochs,
            validation_data = valGen,
            validation_steps = valSteps,
            verbose=1,
            callbacks=[checkpoint])

    return hist
