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

def decompose_patch(high,low,window=100,weight=np.eye(2)):
    H,W = high.shape;
    ica = FastICA(n_components=2,whiten=True,w_init=weight);

    L = np.zeros((H,W,2)); cnt =0;
    for i in range(window,H-window,int(window/2)):
        for j in range(window,W-window,int(window/2)):
            roiHigh = high[i-window:i+window,j-window:j+window];
            roiLow = low[i-window:i+window,j-window:j+window];
            X = np.stack([roiHigh.ravel(), roiLow.ravel()],axis=1);
            S = ica.fit_transform(X); 
            #M = ica.mixing_; 
            L[i-window:i+window,j-window:j+window,:] += S.reshape((2*window,2*window,2)); 
            cnt+=1
    return L/cnt;

import scipy
def decompose_image(high,low,weight):
    H,W = high.shape;
    high = scipy.signal.medfilt(high,3);
    low = scipy.signal.medfilt(low,3);

    #contrast streching
    # Contrast stretching
    #p5, p90 = np.percentile(high, (5, 95))
    #high = exposure.rescale_intensity(high, in_range=(p5, p90))
    #high = exposure.equalize_hist(high)

    #p5, p90 = np.percentile(low, (5, 95))
    #high = exposure.rescale_intensity(low, in_range=(p5, p90))
    #low = exposure.equalize_hist(low);


    ica = FastICA(n_components=2,whiten=True,w_init=weight);
    X = np.c_[high.ravel(), low.ravel()]; 
    idx = np.random.randint(0,X.shape[0],50);
    X_sample = X[idx,:];

    #X = (X-X.mean(axis=0))/X.std(axis=0);
    S = ica.fit_transform(X_sample);
    print(S.shape)
    M = ica.mixing_;
    M = np.linalg.pinv(M);
    M[0,:] = 0;

    X_rec = np.dot(X,M)

    return M, X_rec.reshape((H,W,2)) #S.reshape((H,W,2))

def np_sig(x): return 1/(1+np.exp(-x));
def np_tanh(x): return np.tanh(x);


def sigmoid_gradient_descent(high, low, num_epoch=200, learning_rate=0.0002, w_init=np.eye(2)): #Andrew NG's method
    W = w_init;
    X = np.c_[high.ravel(),low.ravel()];
    X =(X-X.mean(axis=0))/X.std(axis=0);

    for iter in range(num_epoch):
        if (iter%50==0):
            print(iter, W)

        temp = np_sig(np.dot(X,W));
        temp = 1-2*temp;
        W = W + learning_rate*(np.dot(temp.T,X) + np.linalg.inv(W.T));

    S = np.dot(X,W);
    return W,S

def tanh_gradient_descent(high, low, num_epoch=1000, learning_rate=0.0002): #Andrew NG's method
    W = np.eye(2);
    X = np.c_[high.ravel(),low.ravel()];
    #X /=X.std(axis=0);

    for iter in range(num_epoch):
        if (iter%50==0):
            print(iter, W)
        U = np.tanh(np.dot(X,W));
        g = np.linalg.inv(W.T) - (2/len(X)) * np.dot(X.T,U);
        W = W + learning_rate * g;

    S = np.dot(X,W);
    S /= S.std(axis=0);

    return W,S



import sys,os,cv2,glob, dicom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, FastICA, NMF

if __name__=="__main__":

    #air measurement
    high_air = dicom.read_file("../PaloAltoStaticCBCT/Air_120_DCM/Proj_00000.dcm").pixel_array;
    low_air = dicom.read_file("../PaloAltoStaticCBCT/Air_60_DCM/Proj_00001.dcm").pixel_array;
    #assign median for zero measurements
    high_air[high_air==0] = np.median(high_air);
    low_air[low_air==0] = np.median(low_air);


    #get calibration data
    high_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/120_DCM/*.dcm"));
    low_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/60_DCM/*.dcm"));
    #high_imgs =sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/120/*.dcm'));
    #low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/60/*.dcm'));
    high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/120_DCM/*.dcm'));
    low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/60_DCM/*.dcm'));

    roi = (slice(100,800),slice(200,800));
    idx = 400; n= len(high_imgs);
    dim = high_air[roi].shape
    mu_high_soft = 0.144084048584;  mu_high_bone = 1.07143196612;
    mu_low_soft = 0.208418398876; mu_low_bone = 2.17156096576;
    W = np.array([[mu_high_soft, mu_high_bone],[mu_low_soft, mu_low_bone]]);

    while True:

        print("Processing ... {}".format(idx))
        high = dicom.read_file(high_imgs[idx%n]).pixel_array.astype('float32');
        low = dicom.read_file(low_imgs[idx%n]).pixel_array.astype('float32');

        high = pre_process(high/high_air); low = pre_process(low/low_air);
        #high = pre_process(high/high_air); low = pre_process(low/low_air);

        #de = log_subtraction(high,low,w);
        _,X_Rec = decompose_image(high,low,W);
        #X_Rec = decompose_patch(high,low,window=50,weight=W);
        #print(M);
        '''
        if W is None:
            W, L = sigmoid_gradient_descent(high, low);
        else:
            W, L = sigmoid_gradient_descent(high, low, w_init=W);
        '''

        #imgStack = np.hstack([sig_norm(L[:,0].reshape(dim)),sig_norm(L[:,1].reshape(dim))]);
        imgStack = np.hstack([sig_norm(X_Rec[:,:,0]),sig_norm(X_Rec[:,:,1])])
        cv2.imshow("T",imgStack);

        k = cv2.waitKey(0);    
        if k==83: idx+=2;
        elif k==81: idx-=2;
        elif k==119: w+=0.02;
        elif k==115: w-=0.02;
        elif k==27: break
        else: print(k)
    cv2.destroyAllWindows();
