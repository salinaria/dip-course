import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mtimg
from sys import getsizeof

# Read image
img = mtimg.imread('.\ChestXray.tif')
print(getsizeof(img),img.shape)
print(img.dtype)
# Turn image to gray scale
img = img[...,0]
print(getsizeof(img),img.shape)

# Cut left part of body
left_img = img[:,300:]

# Reverse image by X axis 
rev_left_img = left_img[::-1]

# Plot images
figure = plt.figure(tight_layout=True)
(axes1, axes2, axes3) = figure.subplots(nrows=1, ncols=3)

axes1.set_title('Original')
axes1.imshow(img,cmap='gray',vmin=0,vmax=255)

axes2.set_title('Left part')
axes2.imshow(left_img,cmap='gray',vmin=0,vmax=255)

axes3.set_title('X Reversed left part')
axes3.imshow(rev_left_img,cmap='gray',vmin=0,vmax=255)
plt.savefig('./question4')
plt.show()