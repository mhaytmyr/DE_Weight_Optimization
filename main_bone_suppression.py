import pandas as pd
import cv2, glob
import numpy as np
import pydicom, pickle


from config import *
from helper_to_prepare_data import *
from helper_to_train_model import *
from keras.models import load_model

def sig_norm(img,new_max=1,new_min=0):
    alpha = img.mean()
    beta = img.std()
    return (new_max-new_min)/(1+np.exp((img-alpha)/beta))

def log_subtraction(high,low,w=0.72):
    de = high-w*low;
    return sig_norm(de)

def predict_triplet_patch(in1,in2,model):
    H0,W0 = in1.shape;

    if H<H0:
        rowIdx = np.random.randint(0,in1.shape[0]-W);
    else:
        rowIdx = 0;

    if W<W0:
        colIdx = np.random.randint(0,in1.shape[1]-H);
    else:
        colIdx = 0;

    #choose center of image
    #X_in = np.stack([in1[rowIdx:rowIdx+W,colIdx:colIdx+H], in2[rowIdx:rowIdx+W,colIdx:colIdx+H]],axis=2);
    X_in = np.stack([in1, in2],axis=2);
    batch = model.predict(X_in[np.newaxis,:,:,:]);
    return batch[0,:,:,0]


#def plot_triplet_predict(high_imgs,low_imgs,model):
def plot_triplet_predict(valGen,model):

    while True:
        #print("Processing ... {}".format(idx))
        #high = pre_process(pydicom.read_file(high_imgs[idx%n]).pixel_array);
        #low = pre_process(pydicom.read_file(low_imgs[idx%n]).pixel_array);
        data, label = next(valGen);

        outPred = model.predict(data);
        #outPred = predict_triplet_patch(high,low,model);
        
        print(outPred.max(), outPred.min())
        # cv2.imshow("Test",exposure.equalize_hist(outPred));
        cv2.imshow("Test",sig_norm(outPred[0,...]));

        k = cv2.waitKey(0);
        if k==27:
            break
    cv2.destroyAllWindows();

def plot_triplet_data_gen(myGen):

    while True:
        X_in, X_out = next(myGen);
        print(X_in[0,:,:,0].max(),X_in[0,:,:,1].max(),X_out[0,:,:,0].max());
        print(X_in[0,:,:,0].min(),X_in[0,:,:,1].min(),X_out[0,:,:,0].min());
        imgStack = np.hstack([X_in[0,:,:,0],X_out[0,:,:,0]]);
        #imgStack = X_out[0,:,:,0];
        cv2.imshow("image",sig_norm(imgStack));
        k = cv2.waitKey(0)
        if k==27:
            break
        elif k==83:
            continue;

    cv2.destroyAllWindows();

def compare_de(high_imgs,low_imgs,model):
    idx = 300;
    n = len(high_imgs);
    roi = (slice(256,512),slice(256,768))

    while True:
        highImg = pre_process(pydicom.read_file(high_imgs[idx%n]).pixel_array);
        lowImg = pre_process(pydicom.read_file(low_imgs[idx%n]).pixel_array);

        #create original subtraction
        deOrg = log_subtraction(highImg,lowImg);

        #create blended subtraction
        deNeural = sig_norm(predict_triplet_patch(highImg,lowImg,model));

        #plot images
        imgStack = np.hstack([deNeural,sig_norm(highImg)]);
        cv2.imshow("output",imgStack);
        k = cv2.waitKey(0);
        if k==27:
            break
        elif k==83:
            idx+=2;
        elif k==81:
            idx-=2;

    cv2.destroyAllWindows()

def apply_padding(batch):
    new=np.pad(batch[1:-1,1:-1],pad_width=(1,1),mode='edge')
    return new


def save_norm_params(high_files, low_files, de_files, fileName):

    #load images to numpy memory
    imgs = np.array([pydicom.read_file(item).pixel_array for item in high_files]);
    highMean = imgs.mean(axis=0);
    highStd = imgs.std(axis=0);

    #load low images
    imgs = np.array([pydicom.read_file(item).pixel_array for item in low_files]);
    lowMean = imgs.mean(axis=0);
    lowStd = imgs.std(axis=0);

    #load de images
    imgs = np.array([pydicom.read_file(item).pixel_array for item in de_files]);
    softMean = imgs.mean(axis=0);
    softStd = imgs.std(axis=0);


    with h5py.File(fileName, "w") as newFile:
        newFile.create_dataset("high/means",data=highMean, dtype=np.float32);
        newFile.create_dataset("high/vars",data=highStd, dtype=np.float32);
        newFile.create_dataset("low/means",data=lowMean, dtype=np.float32);
        newFile.create_dataset("low/vars",data=lowStd, dtype=np.float32);
        newFile.create_dataset("soft/means",data=softMean, dtype=np.float32);
        newFile.create_dataset("soft/vars",data=softStd, dtype=np.float32);
    print("Norms saved!")

import sys,os
import h5py
#import matplotlib.pyplot as plt

if __name__=="__main__":
    high_air_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/PaloAltoStaticCBCT/Air_120_DCM/*.dcm'));
    low_air_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/PaloAltoStaticCBCT/Air_60_DCM/*.dcm'));

    high_imgs =sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/120_DCM/*.dcm'));
    low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/60_DCM/*.dcm'));
    soft_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/DE/*.dcm'));
    
    #high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_5mm_A0_T0/120_DCM/*.dcm'));
    #low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_5mm_A0_T0/60_DCM/*.dcm'));
    #soft_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_5mm_A0_T0/DE_DCM/*.dcm'));

    fileName = "NORM_STATS.h5"
    #save_norm_params(high_imgs, low_imgs, soft_imgs, fileName)
    
    trainGen = prepare_triplet_data(high_imgs,low_imgs,soft_imgs,
                high_air=None, low_air=None, batch_size=batchSize,rotate=False);

    #high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/120_DCM/*.dcm'));
    #low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/60_DCM/*.dcm'));
    #soft_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/DE_Soft/*.dcm'));
    

    high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_15mm_A0_T0/120_DCM/*.dcm'));
    low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_15mm_A0_T0/60_DCM/*.dcm'));
    soft_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/CBCT_15mm_A0_T0/DE_DCM/*.dcm'));

    valGen = prepare_triplet_data(high_imgs,low_imgs,soft_imgs,high_air=None,low_air=None, batch_size=batchSize,rotate=False)

    #low_imgs = sorted([subdir+'/'+file for subdir,dir,files in os.walk("../PatientData/") if "60kVp" in subdir for file in files]);
    #high_imgs = sorted([subdir+'/'+file for subdir,dir,files in os.walk("../PatientData/") if "120kVp" in subdir for file in files])

    if sys.argv[1]=='train':
        hist = train_patch_cbct(trainGen,valGen);

        #with open('./log/'+modelName, 'wb') as fp:
        #    pickle.dump(hist.history, fp)

        #plt.plot(hist.history['loss'],'r*',hist.history['val_loss'],'g^');
        #plt.show();

    elif sys.argv[1]=='test':
        model = load_model('./checkpoint/'+modelName+'.h5',custom_objects={'mix_loss':mix_loss,'SSIM_cs':SSIM_cs});
        plot_triplet_predict(trainGen,model)
        #compare_de(high_imgs,low_imgs,model);
    elif sys.argv[1]=="plot":
        #plot_triplet_data_gen(valGen)
        plot_triplet_data_gen(trainGen)
