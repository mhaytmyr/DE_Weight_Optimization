import cv2
from skimage import exposure, filters
import numpy as np
import pydicom, h5py
from scipy.signal import medfilt2d
from config import *

def pre_process(img,norm,air_norm=None):
    #img = cv2.medianBlur(img,3);
    if air_norm is None:
        #return img.astype("float32")/img.max();
        return (img-norm["means"])/(norm["vars"]);
        #return np.log(img.astype('float32')+1);
    else:
        imgNorm = img.astype("float32")/(air_norm+1);
        return imgNorm/imgNorm.max();
        #return (img-img.mean())/img.std();
        #return np.log(imgNorm.astype('float32')+1);
        #return exposure.equalize_hist(np.log(img.astype('float32')+1));


def get_air_measurements(high_air,low_air):
    highImgs = [pydicom.read_file(file).pixel_array for file in high_air];
    lowImgs = [pydicom.read_file(file).pixel_array for file in low_air];

    #retrun mean of images
    return np.array(highImgs).mean(axis=0), np.array(lowImgs).mean(axis=0);

def prepare_triplet_data(high_imgs,low_imgs,soft_imgs, high_air=None, low_air=None,batch_size=50,rotate=False):
    """
    Function to return patch of images from 1024 x 768 dataset.
    Idea here is randomly select patches from the image and return triplets for training
    This function will be generator and (H will be multiplies of 1024, W will be multiplies of W)
    Prediction is then done for each patch of the resulting image
    """

    #get triplets of images
    X_in = np.zeros((batch_size,H,W,C));#first and last frame
    X_out = np.zeros((batch_size,H,W,1));
    idx, N = 0, len(low_imgs);

    #get input image shapes
    img = pydicom.read_file(high_imgs[0]).pixel_array;
    H0,W0 = img.shape;

    #if air normalization is given calculate it at the beginning
    if high_air:
        highAir, lowAir = get_air_measurements(high_air,low_air);
    else:
        highAir, lowAir = None,None

    #get image normalization
    #fileName = "NORM_STATS.h5"
    #norm = h5py.File(fileName,"r");
    norm = {"high":{"means":7166.9395,"vars":4486.6328},"low":{"means":4280.3179,"vars":3853.0291},"soft":{"means":32984.023,"vars":11754.251}}

    while True:
        i = 0;
        while i<batch_size:
            #randomly choose starting index
            #idx = np.random.randint(0,N)

            try:
                high = pydicom.read_file(high_imgs[idx%N]).pixel_array;
                low = pydicom.read_file(low_imgs[idx%N]).pixel_array;
                out = pydicom.read_file(soft_imgs[idx%N]).pixel_array;
            except Exception as e:
                #print(e);
                idx+=1;
                continue;

            #randomly choose patches from this sequence
            #we only need to specify xpos ypos
            if H0>H:
                rowpos = np.random.randint(ROWSTART,ROWSTART+MARGIN);
            else:
                rowpos = 0;

            if W0>W:
                colpos = np.random.randint(COLSTART,COLSTART+MARGIN);
            else:
                colpos = 0;

            #high = pre_process(high[rowpos:rowpos+H,colpos:colpos+W],norm["high"],highAir);
            #low = pre_process(low[rowpos:rowpos+H,colpos:colpos+W],norm["low"],lowAir);

            if rotate:
                #randomly rotate image
                #M = cv2.getRotationMatrix2D((H/2,W/2),np.random.randint(0,180),1);
                M = cv2.getRotationMatrix2D((H/2,W/2),np.random.choice([0,90,180,270]),1);

                X_in[i,:,:,0] = cv2.warpAffine(high,M,(W,H));
                X_in[i,:,:,1] = cv2.warpAffine(low,M,(W,H));
                X_out[i,:,:,0] = cv2.warpAffine(pre_process(out[rowpos:rowpos+H,colpos:colpos+W]),M,(W,H));
            else:

                #try:
                X_in[i,:,:,0] = pre_process(high[rowpos:rowpos+H,colpos:colpos+W],norm["high"],highAir);
                X_in[i,:,:,1] = pre_process(low[rowpos:rowpos+H,colpos:colpos+W],norm["low"],lowAir);
                X_out[i,:,:,0] = pre_process(out[rowpos:rowpos+H,colpos:colpos+W],norm["soft"]);
                
            idx+=1; i+=1;
        yield X_in, X_out
