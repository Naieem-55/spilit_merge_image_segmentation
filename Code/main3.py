import numpy as np
import cv2
import os

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
    stack = [(img, 0, 0, img.shape[1], img.shape[0])]  # (image, x, y, width, height)
    segmented_img = np.zeros_like(img)
    while stack:
        img, x, y, w, h = stack.pop()
        if is_homogeneous(img, threshold):
            segmented_img[y:y+h, x:x+w] = img
        else:
            sub_imgs = split_image(img)
            mid_x = w // 2
            mid_y = h // 2
            stack.append((sub_imgs[0], x, y, mid_x, mid_y))  # Top left
            stack.append((sub_imgs[1], x+mid_x, y, w-mid_x, mid_y))  # Top right
            stack.append((sub_imgs[2], x, y+mid_y, mid_x, h-mid_y))  # Bottom left
            stack.append((sub_imgs[3], x+mid_x, y+mid_y, w-mid_x, h-mid_y))  # Bottom right
    return segmented_img

def main():
    # Load the input image
    image_path = r'C:\Users\Hp\OneDrive\Desktop\Work3\Image\panda.jpg'
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

if __name__ == "__main__":
    main()
