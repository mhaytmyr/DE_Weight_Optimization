import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters

def sig_norm(img,new_max=1,new_min=0):
    alpha = img.mean()
    beta = img.std()
    return (new_max-new_min)/(1+np.exp((img-alpha)/beta))

def log_subtraction(high,low,w=0.72):
    de = high-w*low;
    return de

def hist_norm(img):
    return exposure.equalize_hist(img);

def pre_process(img):
    return np.log(img.astype('float32')+1);

def fit_image(high,low,soft,degree=3):

    X_train = np.array([high.ravel(),low.ravel()]);
    y = soft.ravel().T;

    #create image features
    poly = PolynomialFeatures(degree=degree,include_bias=False);
    X_train = poly.fit_transform(X_train.T); 

    #normalize image
    scaler = MinMaxScaler();
    X_norm = scaler.fit_transform(X_train);

    print(X_norm.shape, y.shape)
    #now fit data to features
    reg = LinearRegression();
    reg.fit(X_norm,y);
    #pred = reg.predict(X_norm);

    return reg,scaler

def predict_image(high,low,reg,scaler,degree=3):
    #get pixels of image
    X_train = np.array([high.ravel(),low.ravel()]);

    #create image features
    poly = PolynomialFeatures(degree=degree,include_bias=False);
    X_train = poly.fit_transform(X_train.T); 
    X_norm = scaler.transform(X_train);

    #predict image
    pred = reg.predict(X_norm);

    return pred.reshape(high.shape);


import sys,os,cv2,glob, pydicom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

if __name__=="__main__":


    #air measurement
    high_air = pydicom.read_file("../PaloAltoStaticCBCT/Air_120_DCM/Proj_00000.dcm").pixel_array.astype('float32')
    low_air = pydicom.read_file("../PaloAltoStaticCBCT/Air_60_DCM/Proj_00001.dcm").pixel_array.astype('float32')
    #assign zero measurements to be one
    high_air[high_air==0] = 1
    low_air[low_air==0] = 1

    #polynomial degree
    degree = 3

    #get calibration data
    high_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/120_DCM/*.dcm"));
    low_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/60_DCM/*.dcm"));
    #high_imgs = sorted(glob.glob("../CBCT_15mm_A0_T0/120_DCM/*.dcm"));
    #low_imgs = sorted(glob.glob("../CBCT_15mm_A0_T0/60_DCM/*.dcm"));
    de_imgs = sorted(glob.glob("../CBCT_15mm_A0_T0/DE_DCM/*.dcm"));
        


    idx = 880; n= len(high_imgs);
    w = 0.72;
    while True:
        try:
            print("Processing ...{}, image weight {}".format(idx,w))
            high = pre_process(pydicom.read_file(high_imgs[idx%n]).pixel_array);
            low = pre_process(pydicom.read_file(low_imgs[idx%n]).pixel_array);
            de = log_subtraction(high,low,w);

            #imgStack = np.hstack([sig_norm(de),sig_norm(pred)])
            cv2.imshow("T",sig_norm(de));
        except Exception as e:
            print(e);
            idx+=1;
            continue;

        k = cv2.waitKey(0);    
        if k==83: idx+=2;
        elif k==81: idx-=2;
        elif k==119: w+=0.02;
        elif k==115: w-=0.02;
        elif k==27: break
        else: print(k)
    cv2.destroyAllWindows();

    #fit calibration data
    roi = (slice(256,300),slice(256,300))
    reg,scaler = fit_image(high[roi],low[roi],de[roi],degree);
    #reg,scaler = fit_image(high,low,de,degree);
    print(reg.coef_);

    #now use calibration data for testing
    test_high = sorted(glob.glob("../CBCT_15mm_A0_T0/120_DCM/*.dcm"));
    test_low = sorted(glob.glob("../CBCT_15mm_A0_T0/60_DCM/*.dcm"));
    de_imgs = sorted(glob.glob("../CBCT_15mm_A0_T0/DE_DCM/*.dcm"));    

    #test_high =sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/120_DCM/*.dcm'));
    #test_low = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/60_DCM/*.dcm'));    

    #test_high = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/120_DCM/*.dcm'));
    #test_low = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/60_DCM/*.dcm'));
    #test_de = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/DE_Soft/*.dcm'));

    idx, n = 0, len(test_high);
    shiftX = 0;
    while True:
        try:
            print("Processing ... {}".format(idx))
            high = pre_process(pydicom.read_file(test_high[idx%n]).pixel_array);
            low = pre_process(pydicom.read_file(test_low[idx%n]).pixel_array);
            #de = pre_process(dicom.read_file(test_de[idx%n]).pixel_array);

            #shift image
            low = np.roll(low,shift=shiftX,axis=1);
            pred = predict_image(high,low,reg,scaler,degree);
        except Exception as e:
            print(e);
            idx+=1; continue;
        

        #imgStack = np.hstack([sig_norm(de),sig_norm(pred)])
        cv2.imshow("T",sig_norm(pred));

        k = cv2.waitKey(0);    
        if k==83: idx+=2;
        elif k==81: idx-=2;
        elif k==119: shiftX+=1;
        elif k==115: shiftX-=1;
        elif k==27: break
        else: print(k)
    cv2.destroyAllWindows();
