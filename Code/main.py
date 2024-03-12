import numpy as np
import sys
import cv2

def is_homogeneous(img, threshold, min_size=16):
    # Check if the image dimensions are smaller than the minimum size
    h, w, _ = img.shape
    if h <= min_size or w <= min_size:
        return True

    # Calculate the standard deviation of the image
    std_dev = np.std(img)
    return std_dev < threshold


def split_image(img):
    # Get the dimensions of the input image
    h, w, _ = img.shape

    # Calculate the midpoints
    mid_x = w // 2
    mid_y = h // 2

    # Split the image into four sub-images
    sub_imgs = [
        img[:mid_y, :mid_x],    # Top left
        img[:mid_y, mid_x:],    # Top right
        img[mid_y:, :mid_x],    # Bottom left
        img[mid_y:, mid_x:]     # Bottom right
    ]

    return sub_imgs

sys.setrecursionlimit(10000)
 
 

def segment_image(img, threshold):
    # Check if the image is homogeneous
    if is_homogeneous(img, threshold):
        return img

    # Split the image into four sub-images
    sub_imgs = split_image(img)

    # Initialize a list to store the segmented sub-images
    segmented_sub_imgs = []
 
    # Segment each sub-image separately to avoid deep recursion
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


# Set the threshold (you can adjust this value as needed)
threshold = 50

#Segment the image using the splitting technique
segmented_img = segment_image(img, threshold)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_img)

cv2.destroyAllWindows()


def is_homogeneous(img, threshold, min_size=2):
    # Check if the image dimensions are smaller than the minimum size
    h, w, _ = img.shape
    if h <= min_size or w <= min_size:
        return True

    # Calculate the standard deviation of each sub-image
    sub_imgs = split_image(img)
    std_devs = [np.std(sub_img) for sub_img in sub_imgs]
 
    # Check if all sub-images are homogeneous
    return all(std_dev < threshold for std_dev in std_devs)


def split_image(img):
    # Get the dimensions of the input image
    h, w, _ = img.shape

    # Calculate the midpoints
    mid_x = w // 2 
    mid_y = h // 2

    # Split the image into four sub-images
    sub_imgs = [
        img[:mid_y, :mid_x],    # Top left
        img[:mid_y, mid_x:],    # Top right
        img[mid_y:, :mid_x],    # Bottom left
        img[mid_y:, mid_x:]     # Bottom right
    ]

    return sub_imgs

sys.setrecursionlimit(10000)

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

    # Update the pixels of non-homogeneous sub-images to their mean color value
    for i in range(4):
        if not is_homogeneous(sub_imgs[i], threshold):
            mean_color = np.mean(sub_imgs[i], axis=(0, 1))
            segmented_sub_imgs[i][:] = mean_color 

    # Concatenate the segmented sub-images to form the segmented image
    segmented_img = np.concatenate([
        np.concatenate([segmented_sub_imgs[0], segmented_sub_imgs[1]], axis=1),
        np.concatenate([segmented_sub_imgs[2], segmented_sub_imgs[3]], axis=1)
    ], axis=0)

    return segmented_img



# Load the input image
img = cv2.imread('lena.png', cv2.IMREAD_COLOR)

# Set the threshold (you can adjust this value as needed)
threshold = 25

#segmented_regions = segment_image(img, threshold)

# Combine the segmented regions to form the output image
#output_img = np.zeros_like(img)
#for region in segmented_regions:
#    output_img += region

# Display the output image


# Segment the image using the splitting technique
segmented_img = segment_image(img, threshold)
cv2.normalize(segmented_img, segmented_img, 0, 255, cv2.NORM_MINMAX) 
segmented_img = np.round(segmented_img).astype(np.uint8)

# Display the segmented image
cv2.imshow('Input Image', img)
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



def is_homogeneous(image, threshold):
    """
    Check if the image is homogeneous based on the standard deviation of pixel values.
    """
    std_dev = np.std(image)
    return std_dev < threshold

def split_image(image, threshold):
    """
    Recursively split the image into four sub-images until all sub-images are homogeneous.
    """
    height, width = image.shape[:2]
    
    # Base case: If the image is already homogeneous, return it as is.
    if is_homogeneous(image, threshold):
        return [image]
    
    # Split the image into four sub-images.
    half_height, half_width = height // 2, width // 2
    sub_images = [
        image[:half_height, :half_width],
        image[:half_height, half_width:],
        image[half_height:, :half_width],
        image[half_height:, half_width:]
    ]
    
    # Recursively split each sub-image.
    split_sub_images = []
    for sub_image in sub_images:
        split_sub_images.extend(split_image(sub_image, threshold))
    
    return split_sub_images

sys.setrecursionlimit(10000)

def merge_images(images):
    """
    Merge a list of images into a single image.
    """
    # Determine the size of the output image.
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    total_height = sum(heights)
    total_width = max(widths)
    
    # Create a new image to hold the merged results.
    merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Copy each sub-image into the correct position in the merged image.
    y_offset = 0
    for img in images:
        merged_image[y_offset:y_offset+img.shape[0], :img.shape[1]] = img
        y_offset += img.shape[0]
    
    return merged_image

def main():
    # Load the image.

    image = cv2.imread('lena.png' , cv2.IMREAD_COLOR)
    
    # Ask the user for the threshold.
    threshold = float(input("Enter the threshold for homogeneity: "))
    
    # Perform the segmentation.
    split_images = split_image(image, threshold)
    
    # Merge the split images.
    merged_image = merge_images(split_images)
    
    # Display the original and merged images.
    cv2.imshow('Original Image', image)
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()