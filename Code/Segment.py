import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def splitMergeSegmentation(image, threshold):
    segmented = np.zeros_like(image)
    if np.max(image) - np.min(image) > threshold:
        height, width = image.shape
        halfHeight = height // 2
        halfWidth = width // 2
        
        segmented[0:halfHeight, 0:halfWidth] = splitMergeSegmentation(image[0:halfHeight, 0:halfWidth], threshold)
        segmented[0:halfHeight, halfWidth:] = splitMergeSegmentation(image[0:halfHeight, halfWidth:], threshold)
        segmented[halfHeight:, 0:halfWidth] = splitMergeSegmentation(image[halfHeight:, 0:halfWidth], threshold)
        segmented[halfHeight:, halfWidth:] = splitMergeSegmentation(image[halfHeight:, halfWidth:], threshold)
    else:
        if np.max(image) >= 127:
            segmented[:, :] = np.mean(image)
        else:
            segmented[:, :] = 0
            
    return segmented


image_path = r'C:\Users\Hp\OneDrive\Desktop\Work3\Image\Lena.jpg'
image = cv2.imread(image_path)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = float(input('Enter threshold value : '))
segmentedImage = splitMergeSegmentation(grayImage, threshold)

paddedImage = cv2.copyMakeBorder(segmentedImage, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
averagedImage = np.zeros_like(segmentedImage)

window = 3
padSize = window // 2

for i in range(segmentedImage.shape[0]):
    for j in range(segmentedImage.shape[1]):
        neighborhood = paddedImage[i:i+window, j:j+window]
        averageValue = np.mean(neighborhood)
        averagedImage[i, j] = averageValue

#Save output image into Output folder
output_folder = r'C:\Users\Hp\OneDrive\Desktop\Work3\Output'
output_image_path = os.path.join(output_folder,'Segment.jpg')
cv2.imwrite(output_image_path,segmentedImage)

# Display the original image     
cv2.imshow('Original Image', image)

# Display the segmented image
cv2.imshow('Segmented Image', segmentedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()