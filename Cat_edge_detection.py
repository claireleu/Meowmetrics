from __future__ import print_function
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import csv
import pandas as pd
sigma = 1.0
temp1 = 1/(np.sqrt(2*np.pi)*sigma)
temp2 = np.exp(-2/(2*sigma*sigma))
C13 = temp1*temp2

temp3 = np.exp(-1/(2*sigma*sigma))
C12 = temp1*temp3

temp4 = np.exp(-0/(2*sigma*sigma))

C22 = temp1*temp4

sum = C22 + 4*C13 + 4*C12

C13_normalized = C13/sum
C12_normalized = C12/sum
C22_normalized = C22/sum

# code to crop the coin to decide the pixel size
img_color = cv2.imread('../images/IMG_3822.JPG')
coin = img_color[ 1653:1762,695:810, :]
coin = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 7))
plt.imshow(coin, cmap='gray')
plt.xlabel('Width ($x$)')
plt.ylabel('Height ($y$)')
plt.savefig('../result/coin.jpg')
plt.show()

img = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) #convert from bgr color space to HSV color space
img = np.squeeze(img[:, :, 2]) # Extract the V channel,
img = cv2.GaussianBlur(img,(25, 25),0) #Gaussian blur the V channel
img_copy = img_color.copy()
img_copy[:, :, 0] = img
img_copy[:, :, 1] = img
img_copy[:, :, 2] = img

cv2.rectangle(img_copy, (920,1950), (1020,2050), (255, 0, 255), 4) # Show the rectangle at the edge

plt.figure(figsize=(10, 7))
plt.imshow(img_copy, cmap='gray')
plt.xlabel('Width ($x$)')
plt.ylabel('Height ($y$)')
plt.savefig('../result/3822_with_gaussian.jpg')
plt.show()

# Show the Gaussian blurred image
plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray')
plt.xlabel('Width ($x$)')
plt.ylabel('Height ($y$)')
plt.savefig('../result/3822_gaussian.jpg')
plt.show()

#Show the cropped edge
cropped = img[1950:2050,920:1020]
DF = pd.DataFrame(cropped)
# DF.to_csv("cropped_with_gaussian.csv")

plt.figure(figsize=(10, 7))
plt.imshow(cropped, cmap='gray')
plt.xlabel('Width ($x$)')
plt.ylabel('Height ($y$)')
plt.savefig('../result/3822_cropped_gaussian.jpg')
plt.show()

ksize = 3
#Gradient in the horizontal direction
gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
#Gradient in the vertical direction
gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

# the gradient magnitude images are now of the floating point data
# type, so we need to take care to convert them back a to unsigned
# 8-bit integer representation so other OpenCV functions can operate
# on them and visualize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

#Gradient in the horizontal direction
DF = pd.DataFrame(gX[1950:2050,920:1020])
# DF.to_csv("../result/gradient_horizontal.csv")

#Gradient in the vertical direction
DF = pd.DataFrame(gY[1950:2050,920:1020])
# DF.to_csv("../result/gradient_vertical.csv")

#Gradient in both directions
combined = np.uint8(np.sqrt(gX.astype(float)**2 + gY.astype(float)**2))
gX_Y = combined

plt.figure(figsize=(10, 7))
plt.plot(np.arange(0, 100, 1),gX[1950+96,920:1020], label = 'Derivatives of row 96 without Gaussian blur')
plt.xlabel('Width')
plt.ylabel('Intensity')
# plt.legend()
plt.legend(loc='upper right')
plt.savefig('../result/3822_no_gaussian_row_96_derivative.jpg')
plt.show()

# show the gradients
plt.figure(figsize=(10, 7))
plt.subplot(2,2,1),plt.imshow(cropped,cmap = 'gray')
plt.title('Raw image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(gX[1950:2050,920:1020],cmap = 'gray')
plt.title('Gradient in Horizontal Direction'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(gY[1950:2050,920:1020],cmap = 'gray')
plt.title('Gradient in Vertical Direction'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(gX_Y[1950:2050,920:1020],cmap = 'gray')
plt.title('Combined Gradient'), plt.xticks([]), plt.yticks([])

plt.savefig("../result/3822_cropped_with_gaussian_gradient.jpg")
plt.show()
# combine the gradient representations into a single image
# combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

#Threshold the combined gradient to eliminate the areas with low variation
combined[combined<10] = 0
combined[combined>=10] = 255

#Process the image to connect the broken connected areas
dilation_shape = cv2.MORPH_ELLIPSE
dilatation_size = 3

element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

combined_dilated = cv2.dilate(combined, element)

dilatation_size =1
element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
combined_eroded = cv2.erode(combined_dilated, element)

#show the rough edge
plt.figure(figsize=(10, 7))
plt.subplot(2,2,1),plt.imshow(gX,cmap = 'gray')
plt.title('Gradient in Horizontal Direction'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(gY,cmap = 'gray')
plt.title('Gradient in Vertical Direction'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(combined,cmap = 'gray')
plt.title('Combined Gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(combined_eroded,cmap = 'gray')
plt.title('Post-processed Combined Gradient'), plt.xticks([]), plt.yticks([])
plt.show()

contours, hierarchy = cv2.findContours(combined_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_area = 0
largest_contour_index = 0
for i, contour in enumerate(contours ): # iterate through each contour.
    area = cv2.contourArea( contours[i] );  #  Find the area of contour
    if area > largest_area:
        largest_area = area;
        largest_contour_index = i;               #Store the index of largest contour
        bounding_rect = cv2.boundingRect( contours[i] ); # Find the bounding rectangle for biggest contour

cv2.drawContours( img_color, contours,largest_contour_index, ( 255, 0, 0 ), 5 ); # Draw the largest contour using previously stored index.

# show the images
cv2.imshow("Result", img_color)
cv2.imwrite('../result/cat_outline.png', img_color)

cv2.waitKey(0)

print(0)