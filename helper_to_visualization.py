import numpy as np
import matplotlib.pyplot as plt
import cv2,dicom
from config import *
from helper_to_prepare_data import pre_pocess

def plot_random_img(X_in, X_pred, X_rot45, number_images=50, time_delay=3000):
    max_N = X_in.shape[0];

    #display for visualization
    for _ in range(number_images):
        idx = np.random.randint(max_N)
        img_tot = np.hstack([X_in[idx], X_pred[idx], X_rot45[idx]])
	cv2.imshow("Test",img_tot)
        cv2.waitKey(time_delay)
        cv2.destroyAllWindows()

def sig_norm(img,new_max=255,new_min=0):
    alpha = img.mean()
    beta = img.std()
    return (new_max-new_min)/(1+np.exp((img-alpha)/beta))

def linear_norm(img,new_max=255,new_min=0):
    return new_min+(img-img.min())*(img.max()-img.min())/(new_max-new_min)

def get_cnn_interpolation(img1,img2,model,alpha=0.5):
    #split autoencoder
    encoder = model.layers[0]
    decoder = model.layers[1]

    # Create micro batch
    X = np.concatenate([img1[np.newaxis,:,:,np.newaxis], img2[np.newaxis,:,:,np.newaxis]],axis=0)

    # Compute latent space projection
    latentX = encoder.predict(X.astype('float32')/(2**16-1))
    latentStart, latentEnd = latentX
    latentInter = np.array(latentStart*(1-alpha) + latentEnd*alpha)
    res = decoder.predict(latentInter[np.newaxis,:,:,:])*(2**16-1) #rescale back to original

    #return spatial dimensions
    return res[0,:,:,0]


def warp_flow(img, flow, back_proj=True):

	height, width = flow.shape[:2]
	R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
	if back_proj:
		pixel_map = R2+flow;
	else:
		pixel_map = R2-flow;
	res = cv2.remap(img.astype('float32'), pixel_map.astype('float32'),None, cv2.INTER_CUBIC, borderMode =cv2.BORDER_REPLICATE )
	return res

def get_interlace_and_opticalflow(prev_img, curr_img):

	flow = cv2.calcOpticalFlowFarneback(prev_img.astype('float32')/(2**16-1), 
		curr_img.astype('float32')/(2**16-1), None, 0.5, 3, 15, 3, 5, 1.2, 0)

	#propogate image
	img_fopt = warp_flow(curr_img.astype('float32'),flow,back_proj=True)
	img_bopt = warp_flow(prev_img.astype('float32'),flow,back_proj=False)
	

	img_interlace = np.zeros_like(prev_img)
	img_interlace[:,0::2] = prev_img[:,0::2];
	img_interlace[:,1::2] = curr_img[:,1::2];

	# hsv = np.zeros_like(prev_img)
	# hsv[:,1] = 255
	# mag, ang = cv2.cartToPolar(flow[:,0], flow[:,1])
	# hsv[:,0] = ang*180/np.pi/2
	# hsv[:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	# bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	return img_bopt, img_interlace


def plot_de_cbct(high_images, low_images, model, H,W, time_delay=300):
	max_N = len(high_images);
	

	de_rgb = np.empty((H, W*2, 3), dtype=np.uint8)
	images = np.empty((max_N, H,W*2,3));
	cv2.namedWindow('Dual CBCT',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Dual CBCT', 1600,800)

	for idx in range(1,max_N-1):
		high_img = dicom.read_file(high_images[idx]).pixel_array;
		low_img = dicom.read_file(low_images[idx]).pixel_array;
		de_img = np.uint8(sig_norm(np.log(high_img+1)-0.7*np.log(low_img+1)));
		
		#compute interpolated image
		low_prev = dicom.read_file(low_images[idx-1]).pixel_array;
		#low_opt, low_inter = get_interlace_and_opticalflow(low_prev, low_img)

		#de_opt = np.uint8(sig_norm(np.log(high_img+1)-0.7*np.log(low_opt+1)));
		#de_inter = np.uint8(sig_norm(np.log(high_img+1)-0.7*np.log(low_inter+1)));


		#fit model on given images
		model.compile(optimizer='adam',loss='mean_absolute_error')
		model.fit(low_img[np.newaxis,:,:,np.newaxis].astype('float32')/(2**16-1), 
			low_prev[np.newaxis,:,:,np.newaxis].astype('float32')/(2**16-1), 
			batch_size=10, epochs=10, verbose=0)

		low_middle = get_cnn_interpolation(low_img, low_prev, model, alpha=0.5)
		de_new = np.uint8(sig_norm(np.log(high_img+1)-0.7*np.log(low_middle+1)));

		#combine two images
		de_stack = np.hstack([de_img, de_new])
		de_rgb[:,:,0] = de_stack; de_rgb[:,:,1] = de_stack; de_rgb[:,:,2] = de_stack;

		if idx%10==0:
			print("Writing image ...",idx)

			cv2.imshow("Dual CBCT", de_rgb)
			cv2.waitKey(time_delay);
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break


		images[idx,:,:,:] = de_rgb;

	return images
	
def plot_random_transformation(X_in, X_rot45, model, number_images=50, time_delay=3000):
	max_N = X_in.shape[0];
	nbSteps = 5;
	alphaValues = np.linspace(0, 1, nbSteps)

	#de-attach encoder and decoder
	encoder = model.layers[0];
	decoder = model.layers[1];

	#display for visualization
	for _ in range(number_images):
		idx = np.random.randint(max_N)
		img1 = X_in[idx];
		img2 = X_rot45[idx];

		X = np.concatenate([img1[np.newaxis,:,:,:], img2[np.newaxis,:,:,:] ],axis=0)

		# Compute latent space projection
		latentX = encoder.predict(X)
		latentStart, latentEnd = latentX

		#Linear interpolation
		vectors = []
	
		for alpha in alphaValues:
			# Latent space interpolation
			vector = np.array(latentStart*(1-alpha) + latentEnd*alpha)
			vectors.append(vector)

		#now decode final image
		recon = decoder.predict(np.array(vectors))

		#convert to image for visualization
		img_tot = np.hstack([item for item in recon])

		#show reco images
		cv2.imshow("Test",img_tot)
		cv2.waitKey(time_delay)
		cv2.destroyAllWindows()



def plot_train_output(T_org,T_rot,model,hist):

    y = model.predict(T_org)

    fig,axes = plt.subplots(ncols=2, nrows=2,figsize=(12,8))
    ax = axes.ravel()
    ax[0].imshow(T_org[0,:,:,0],cmap="gray")
    ax[1].imshow(y[0,:,:,0],cmap="gray")
    ax[2].plot(hist.history['loss'],"r*")
    ax[2].plot(hist.history['val_loss'],"g.")
    ax[3].imshow(T_rot[0,:,:,0],cmap="gray")
    plt.legend()
    plt.show()

def plot_image_transformation(img1,img2,encoder,decoder,nbSteps):
	
	# Create micro batch
	X = np.concatenate([img1, img2],axis=0)

	# Compute latent space projection
	latentX = encoder.predict(X)
	latentStart, latentEnd = latentX

	fig, axes = plt.subplots(ncols=nbSteps, figsize=(15,12))
	ax = axes.ravel();

	#Linear interpolation
	vectors = []
	alphaValues = np.linspace(0, 1, nbSteps)
	for alpha in alphaValues:
	    # Latent space interpolation
	    vector = np.array(latentStart*(1-alpha) + latentEnd*alpha)
	    vectors.append(vector)
	    
	recon = decoder.predict(np.array(vectors))

	for idx in range(recon.shape[0]):
	    ax[idx].imshow(recon[idx,:,:,0],cmap="gray")
