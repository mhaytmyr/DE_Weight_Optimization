import numpy as np
import pydicom, glob
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

mu_high_soft = 0.144084048584;  mu_high_bone = 1.07143196612;
mu_low_soft = 0.208418398876; mu_low_bone = 2.17156096576;

def auto_encoder_network(high,low,latentNode=2):
    #shape of input image
    H,L = high.shape;
    nepochs = 500;

    #create features placeholder
    tf_X = tf.placeholder(dtype=tf.float32, shape=[None,latentNode]);

    #initialize weighting matrix
    tf_W = tf.Variable(tf.constant([[mu_high_bone, mu_high_soft],[mu_low_bone, mu_low_soft]]));
    tf_Z = tf.matmul(tf_X,tf_W);
    tf_XRec = tf.matmul(tf_Z,tf.transpose(tf_W));

    #define loss
    tf_loss = tf.reduce_mean(tf.abs(tf_X-tf_XRec))+tf.reduce_mean(tf.nn.sigmoid(tf_Z));

    #define model
    train_op = tf.train.AdamOptimizer(learning_rate=1e-02).minimize(tf_loss);
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-03).minimize(tf_loss);

    #create X_in
    X_in = np.c_[high.ravel(),low.ravel()];
    np.random.shuffle(X_in);

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        
        for epoch in range(nepochs):
            W,Z,_,loss = sess.run([tf_W,tf_Z,train_op,tf_loss],feed_dict={tf_X:X_in});
            if epoch%50==0:
                print("Error in epoch {0} is {1}".format(epoch,loss));
                print(W);

    #get the inverse of weights
    W = np.linalg.inv(W);
    X_in = np.vstack([high.ravel(),low.ravel()]);
    X_rec = np.dot(X_in.T,W); 
    X_rec = X_rec.reshape((H,L,2));
    print(Z.shape)
    Z_out = np.stack([Z[:,0].reshape((H,L)), Z[:,1].reshape((H,L))],axis=-1);    

    return W,Z_out

def auto_associate_network(high,low,latentNode=2):
    #shape of input image
    H,L = high.shape;

    #get dimentions of data
    N = np.prod(high.shape);

    #create placeholder
    tf_H = tf.placeholder(dtype=tf.float32,shape=[None,1]);
    tf_L = tf.placeholder(dtype=tf.float32,shape=[None,1])
    
    #create forward pass
    #tf_X = tf.Variable(tf.random_normal(shape=[1,latentNode]));    
    tf_X = tf.Variable(tf.constant([[0.68185264,  1.1894778]]));    
    tf_mu_L = tf.Variable(tf.constant([[mu_low_bone], [mu_low_soft]])); 
    tf_L_pred = tf.matmul(tf.nn.sigmoid(tf.matmul(tf_H,tf_X)),tf_mu_L);
    
    #create backward pass
    # tf_Y = tf.Variable(tf.random_normal(shape=[1,latentNode]));    
    tf_Y = tf.Variable(tf.constant([[0.68403989,  0.90899187]]));    
    tf_mu_H = tf.Variable(tf.constant([[mu_high_bone], [mu_high_soft]])); 
    tf_H_pred = tf.matmul(tf.nn.sigmoid(tf.matmul(tf_L,tf_Y)),tf_mu_H);
    
    #add constrains
    const1 = tf.tensordot(tf_X,tf_mu_H,axes=1);
    const2 = tf.tensordot(tf_Y,tf_mu_L,axes=1);

    #create weight matrix
    tf_W = tf.stack([tf.transpose(tf_mu_H), tf.transpose(tf_mu_L)],axis=1);
    
    #define latent loss
    #latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean)-tf.exp(self.z_log_sigma_sq), 1)

    #define loss
    tf_loss = tf.reduce_mean(tf.pow(tf_H-tf_H_pred,2))+tf.reduce_mean(tf.pow(tf_L-tf_L_pred,2))+\
        tf.abs(tf.matrix_determinant(tf_W));
        #tf.abs(tf.add(tf.reduce_sum(const1),-1)) + tf.abs(tf.add(tf.reduce_sum(const2),-1)) +

    #define model
    train_op = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(tf_loss);
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-03).minimize(tf_loss);

    #create X_in
    #X_in = np.c_[high.ravel(),low.ravel()];

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        
        for epoch in range(500):
            mu_H,mu_L,_,loss,X,Y = sess.run([tf_mu_H,tf_mu_L,train_op,tf_loss,tf_X,tf_Y],
                feed_dict={tf_H:high.ravel()[:,np.newaxis],tf_L:low.ravel()[:,np.newaxis]});
            if epoch%50==0:
                W = np.vstack([mu_H.T,mu_L.T]);
                print("Error in epoch {0} is {1}".format(epoch,loss));
                print(W);
    print(X);
    print(Y);
    #get the inverse of weights
    #W = np.linalg.inv(W);
    #print(W)
    #W = W[0];
    X_in = np.vstack([high.ravel(),low.ravel()]);
    X_rec = np.dot(X_in.T,W); 
    X_rec = X_rec.reshape((H,L,2));


    #S = S.reshape((H,L,2))
    return W,X_rec

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__=="__main__":
    #air measurement
    high_air = pydicom.read_file("../PaloAltoStaticCBCT/Air_120_DCM/Proj_00000.dcm").pixel_array;
    low_air = pydicom.read_file("../PaloAltoStaticCBCT/Air_60_DCM/Proj_00001.dcm").pixel_array;
    #assign median for zero measurements
    #high_air[high_air==0] = np.median(high_air);
    #low_air[low_air==0] = np.median(low_air);

    #get calibration data
    high_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/120_DCM/*.dcm"));
    low_imgs = sorted(glob.glob("../PaloAltoStaticCBCT/60_DCM/*.dcm"));

    high_imgs =sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/120/*.dcm'));
    low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/60/*.dcm'));
    high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/120_DCM/*.dcm'));
    low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/BadenCBCT/60_DCM/*.dcm'));

    roi = (slice(200,500),slice(400,700));
    idx = 100; n= len(high_imgs);
    dim = high_air.shape

    
    while True:
        #load some test data
        high = pre_process(pydicom.read_file(high_imgs[idx%n]).pixel_array.astype('float32'));
        low = pre_process(pydicom.read_file(low_imgs[idx%n]).pixel_array.astype('float32'));

        #W,L = auto_associate_network(high,low,latentNode=2);
        W,L = auto_encoder_network(high,low,latentNode=2);
  
        #imgStack = np.hstack([L[:,0].reshape(dim),L[:,1].reshape(dim)]);
        imgStack = np.hstack([sig_norm(L[:,:,0]),sig_norm(L[:,:,1])])
        #cv2.imshow("T",sig_norm(L[:,0].reshape(dim)));
        cv2.imshow("T",imgStack);


        k = cv2.waitKey(0);
        if k==83: idx+=2;
        elif k==81: idx-=2;
        elif k==119: w+=0.02;
        elif k==27: break
        else: print(k)
    cv2.destroyAllWindows();
