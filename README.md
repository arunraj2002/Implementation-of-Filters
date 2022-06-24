# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary packages numpy,cv2 and matplotlib and save the image to perform Image filtering.

### Step2:
Use the syntax cv2.filter2D() to perform filtering.

### Step3:
For Average filter use the syntax = kernel=np.ones((9,9),np.float32)/81 .

### Step4:
For Weighted average filter use the syntax = kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])/16 .

### Step5:
For Gaussian filter use the syntax = gaussian_blur=cv2.GaussianBlur(src=image2,ksize=(11,11),sigmaX=0,sigmaY=0).

### Step6:
For Median filter use the syntax = median=cv2.medianBlur(src=image2,ksize=11).

### Step7:
For Laplacian kernel filter use the syntax = kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]]).

### Step8:
For Laplacian operator use the syntax = lap_operator=cv2.Laplacian(image2,cv2.CV_64F).

### Step9:
Run the program by using the syntax , and print the output.

## Program:
### Developed By   : R ARUNRAJ
### Register Number: 212220230004
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("bts.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
```

### 1. Smoothing Filters
i) Using Averaging Filter
```Python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
iii) Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
iv) Using Median Filter
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```
ii) Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters

i) Using Averaging Filter

![Avg filter](https://user-images.githubusercontent.com/75235747/168127970-b733f679-527a-41f6-a738-5e2055b0cc05.JPG)

ii) Using Weighted Averaging Filter

![weighted Avg](https://user-images.githubusercontent.com/75235747/168128053-526260f5-8de8-40b8-869d-2c361c96965f.JPG)

iii) Using Gaussian Filter

![Guassian](https://user-images.githubusercontent.com/75235747/168128103-de9e3967-7e3a-4c22-85cc-70229e8127ec.JPG)

iv) Using Median Filter

![Median](https://user-images.githubusercontent.com/75235747/168128134-2b3800b9-3a4e-45bb-a3b0-538cd9a2a662.JPG)

### 2. Sharpening Filters

i) Using Laplacian Kernal

![kernal](https://user-images.githubusercontent.com/75235747/168128340-cf10345e-7419-4c91-a255-e3ce80796870.JPG)

ii) Using Laplacian Operator

![laplaacian](https://user-images.githubusercontent.com/75235747/168128350-2106f79b-5524-4261-804d-5b3712f9970f.JPG)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
