import numpy as np
import dicom
from PIL import ImageTk as pil_imagetk
from PIL import Image as pil_image

from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from skimage.filters import threshold_otsu, threshold_local
import cv2

from tkinter import *

class SliderPosition(Tk):
	def __init__(self,master_window, low_files, high_files):
		self.x = None
		self.y = None
		self.weight = 1
		self.window = master_window
		self.low_img = low_files
		self.high_img = high_files


		self.__init_panel()

	def __init_panel(self):
		#initiate panel
		de_img = np.exp(-(np.log(self.high_img)-0.01*np.log(self.low_img)))
		self.de_img_norm = (2**8-1)*(de_img-de_img.min())/(de_img.max()-de_img.min())
		self.de_img_obj = pil_imagetk.PhotoImage(pil_image.fromarray(self.de_img_norm))
		self.panel = Label(self.window, image = self.de_img_obj)

		#The Pack geometry manager packs widgets in rows or columns.
		self.panel.pack(side = "top", expand="no", fill="none")

	def update_x(self,val):
		self.x = val
		#print(self.print_position())
	def update_y(self,val):
		self.y = val
		#print(self.print_position())
	def print_position(self):
		return self.x, self.y

	def apply_log_subtraction(self,val):
		self.weight = float(val)/100

		de_img_norm = self.low_img.copy()
		#print("Low image ",self.low_img.shape)
		#print("High image ",self.high_img.shape)
		print(self.weight)

		de_img = np.exp(-(np.log(self.high_img)-self.weight*np.log(self.low_img)))
		self.de_img_norm = (2**8-1)*(de_img-de_img.min())/(de_img.max()-de_img.min())

		# self.de_img_norm = (2**16-1)*(de_img-de_img.min()).astype(np.uint32)
		#update image on panel
		self.update_panel_image()
		# return self.de_img_norm

	def update_panel_image(self):
		self.de_img_obj = pil_imagetk.PhotoImage(pil_image.fromarray(self.de_img_norm))
		self.panel.configure(image=self.de_img_obj)
		self.panel.image = self.de_img_obj


# high_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/120_kVp/DS-1-120.dcm").pixel_array
high_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/EM_05092014/120kVp/H_001.dcm").pixel_array
high_img = cv2.bitwise_not(high_img)

# low_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/60_kVp/DS-1-60.dcm").pixel_array
low_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/EM_05092014/60kVp/L_001.dcm").pixel_array
low_img = cv2.bitwise_not(low_img)


#This creates the main window of an application
window = Tk()
window.title("Dual Energy Image")
window.geometry("1124x868")
window.configure(background='grey')


#initilize class to store widget positions
slider = SliderPosition(window,low_img,high_img)

w1 = Scale(window, from_=0, to=100, orient = HORIZONTAL, tickinterval=10, length = 500, command=slider.apply_log_subtraction)
w1.pack(side = "bottom", fill="none")


#w1 = Scale(window, from_=0, to=45, orient = VERTICAL, tickinterval=5, length = 300, command=slider.update_x)
#w1.pack(side = "right", fill="none")

#w2 = Scale(window, from_=0, to=200, orient = HORIZONTAL, tickinterval=10, length=600, command=slider.update_y)
#w2.pack(side = "bottom", fill="none")
#Button(master,text="Show", command = show_value).pack()

# #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
# img = pil_imagetk.PhotoImage(pil_image.fromarray(slider.apply_log_subtraction(50)))

# #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
# panel = Label(window, image = img)
# #The Pack geometry manager packs widgets in rows or columns.
# panel.pack(side = "top", expand="no", fill="none")

window.mainloop()



