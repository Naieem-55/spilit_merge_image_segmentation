import numpy as np
import cv2

def is_homogeneous(img, threshold):
    # Check if the standard deviation of pixel values in the image is less than the threshold
    std_dev = np.std(img)
    return np.isnan(std_dev) or std_dev < threshold

def split_image(img):
    # Split the image into four sub-images
    h, w, _ = img.shape
    mid_x = w // 2
    mid_y = h // 2
    sub_imgs = [
        img[:mid_y, :mid_x],    # Top left
        img[:mid_y, mid_x:],    # Top right
        img[mid_y:, :mid_x],    # Bottom left
        img[mid_y:, mid_x:]     # Bottom right
    ]
    return sub_imgs

def segment_image(img, threshold):
    # Check if the image is homogeneous
    if is_homogeneous(img, threshold):
        return img

    # Split the image into four sub-images
    sub_imgs = split_image(img)

    # Initialize a list to store the segmented sub-images
    segmented_sub_imgs = []

    # Segment each sub-image separately
    for i in range(4):
        segmented_sub_img = segment_image(sub_imgs[i], threshold)
        segmented_sub_imgs.append(segmented_sub_img)

    # Concatenate the segmented sub-images to form the segmented image
    segmented_img = np.concatenate([
        np.concatenate([segmented_sub_imgs[0], segmented_sub_imgs[1]], axis=1),
        np.concatenate([segmented_sub_imgs[2], segmented_sub_imgs[3]], axis=1)
    ], axis=0)

    return segmented_img


# Load the input image
image_path = r'C:\Users\Hp\OneDrive\Desktop\Work3\Image\Lena.jpg' 
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Ask the user for the threshold
threshold = float(input("Enter the threshold for homogeneity: "))

# Segment the image using the splitting technique
segmented_img = segment_image(img, threshold)

# Display the original image
cv2.imshow('Original Image', img)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()