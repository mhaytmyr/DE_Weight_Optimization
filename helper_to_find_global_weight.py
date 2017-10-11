
import numpy as np
import dicom
import matplotlib.pyplot as plt


from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from skimage.filters import threshold_otsu, threshold_local
import cv2

import tensorflow as tf

def sigmoid_normalization(img,min_new=0,max_new=2**16-1,power=-1):
    min_old, max_old = min(img.flat), max(img.flat)
    alpha = np.std(img.flat) # defines the width of the input intensity range,
    beta = np.mean(img.flat) #defines the intensity around which the range is centered

    img_norm = (max_new-min_new)*(1+np.exp(-(img-beta)/alpha))**power+min_new
    return img_norm

#open files and select region of interest
def choose_region_of_interest(high_img):
    #get image pixels

    #first normalize image so that everything is visible
    img = sigmoid_normalization(high_img)

    block_size = 65
    adaptive_thresh = threshold_local(img, block_size, offset=0)
    binary_adaptive = img*(img > adaptive_thresh)

    #now call cv2
    soft_region = cv2.selectROI("Select Soft Region",binary_adaptive)
    tissue = img[soft_region[1]:soft_region[1]+soft_region[3], soft_region[0]:soft_region[0]+soft_region[2]]
    cv2.destroyAllWindows()

    bone_region = cv2.selectROI("Select Bone Region",binary_adaptive)
    bone = img[bone_region[1]:bone_region[1]+bone_region[3], bone_region[0]:bone_region[0]+bone_region[2]]
    cv2.destroyAllWindows()

    #now plot tissue and bone images for testing on original image
    # fig,axes = plt.subplots(ncols=2,figsize=(12,8))
    # ax =axes.ravel()
    # ax[0].imshow(tissue,cmap="gray")
    # ax[0].set_title("Soft")

    # ax[1].imshow(bone)
    # ax[1].set_title("Bone",cmap="gray")
    # plt.show()

    return soft_region, bone_region

import cv2

def global_weight_optimization(soft_region,bone_region,high_img,low_img):
    M,N = high_img.shape

    tf_L = tf.placeholder(tf.float32,[M,N,1])
    tf_H = tf.placeholder(tf.float32,[M,N,1])

    tf_w = tf.Variable(tf.truncated_normal(shape=[1]))

    tf_DE = tf.subtract(tf.log(tf_H),tf_w*tf.log(tf_L))
    #tf_DE_norm  = (2**16-1)*tf.exp(-tf_DE)/tf.reduce_max(tf_DE)
    tf_DE_norm  = (2**16-1)*(tf.exp(-tf_DE))


    tf_soft_roi = tf.image.crop_to_bounding_box(tf_DE_norm,
                        offset_height=soft_region[1],target_height=soft_region[3],
                        offset_width=soft_region[0],target_width=soft_region[2],
                        )

    tf_bone_roi = tf.image.crop_to_bounding_box(tf_DE_norm,
                        offset_height=bone_region[1],target_height=bone_region[3],
                        offset_width=bone_region[0],target_width=bone_region[2],
                        )

    loss = (tf.reduce_mean(tf_soft_roi)-tf.reduce_mean(tf_bone_roi))
    trainer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    # trainer = tf.train.RMSPropOptimizer(learning_rate=0.1,decay=0.9,centered=True).minimize(loss)
    # clip_op = tf.assign(tf_w, tf.clip_by_value(tf_w, 0, np.inf))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(400):
            _, de_img_norm ,tmp = sess.run([trainer,tf_DE_norm,loss],
                    feed_dict={tf_H:high_img[:,:,np.newaxis],tf_L:low_img[:,:,np.newaxis]})

            soft_roi, bone_roi, de_img = sess.run([tf_soft_roi,tf_bone_roi, tf_DE],
                   feed_dict={tf_H:high_img[:,:,np.newaxis],tf_L:low_img[:,:,np.newaxis]})

            if epoch%10==0:
                print("Loss ",tmp," in epoch ",epoch)
        print("Optimal weight found ",sess.run(tf_w))

    fig, axes =plt.subplots(ncols=2, figsize=(12,8))
    ax = axes.ravel()

    # ax[0].imshow(soft_roi[:,:,0],cmap="gray")
    # ax[1].imshow(bone_roi[:,:,0],cmap="gray")
    ax[0].imshow(de_img_norm[:,:,0],cmap="gray")
    ax[1].imshow(sigmoid_normalization(de_img[:,:,0]),cmap="gray")
    plt.show()
    # cv2.imshow("image",de_img[:,:,0])
    # cv2.waitKey(0)


'''
Don't change following. This is how you actually calculate DE
'''
def plot_global_weight(hi,lo,weight=0.5):
    de_img = (2**16-1)*np.exp(-(np.log(hi)-0.5*np.log(lo)))

    plt.imshow(de_img,cmap="gray")
    plt.show()
    return 0

'''
This helper function to find global weights using ROI mehtod.
When you call this function it prompts to open an image and asks to selecet region of image
Using region of interest perorms weight subtraction

'''
import matplotlib

#first prompt GUI to choose image of interest
if __name__=="__main__":
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    low_file = askopenfilename(title="Choose low energy files") # show an "Open" dialog box and return the path to the selected file
    print(low_file)
    high_file = askopenfilename(title="Choose high energy files")
    print(high_file)

    high_img = dicom.read_file(high_file).pixel_array
    high_img = cv2.bitwise_not(high_img)

    low_img = dicom.read_file(high_file).pixel_array
    low_img = cv2.bitwise_not(low_img)

    # plot_global_weight(hi_tmp,lo_tmp)


    #extract region of interest from GUI
    soft_region,bone_region = choose_region_of_interest(high_img)
    print(soft_region)
    print(bone_region)

    global_weight_optimization(soft_region,bone_region,high_img,low_img)





