import pandas as pd
import cv2,glob
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras.backend import tf as ktf


from config import *
from helper_to_prepare_data import *
from helper_to_visualization import *



high_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/120/*.dcm'));
low_imgs = sorted(glob.glob('/home/upsilon/Desktop/DualEnergyProject/MotionPhantomAnalysis/CBCT_A0_T0/60/*.dcm'));

#get all images
#X_start, X_end = prepare_cone_beam_data(high_imgs, low_imgs, H,W, batch_size=50)
loaded_model = load_model(model_name+".h5",custom_objects={'ktf':ktf})
#plot_random_transformation(X_start, X_end, loaded_model,number_images=20)

image_seq = plot_de_cbct(high_imgs[:100], low_imgs[:100], loaded_model, H,W, time_delay=500)


import moviepy.editor as mpy
clip = mpy.ImageSequenceClip(list(image_seq), fps=5)
clip.write_videofile("output.mp4",audio=False)

