import numpy as np
import cv2
import os

def split_and_merge(img):
        
    if img.shape[0] <= 2 and img.shape[1] <= 2:
        return img
    
    blue, green, red = cv2.split(img)
    
    if np.std(blue) < threshold and np.std(green) < threshold and np.std(red) < threshold:
        blue.fill(np.mean(blue))
        green.fill(np.mean(green))
        red.fill(np.mean(red))
        
        img = cv2.merge((blue, green, red))
        return img
    
    halfx = img.shape[0] // 2
    halfy = img.shape[1] // 2
    
    r0b = blue[0 : halfx, 0: halfy]
    r1b = blue[0 : halfx, halfy : img.shape[1]]
    r2b = blue[halfx : img.shape[0], 0: halfy]
    r3b = blue[halfx : img.shape[0], halfy : img.shape[1]]
    
    r0g = green[0 : halfx, 0: halfy]
    r1g = green[0 : halfx, halfy : img.shape[1]]
    r2g = green[halfx : img.shape[0], 0: halfy]
    r3g = green[halfx : img.shape[0], halfy : img.shape[1]]
    
    r0r = red[0 : halfx, 0: halfy]
    r1r = red[0 : halfx, halfy : img.shape[1]]
    r2r = red[halfx : img.shape[0], 0: halfy]
    r3r = red[halfx : img.shape[0], halfy : img.shape[1]]
    
    r0 = split_and_merge(cv2.merge((r0b, r0g, r0r)))
    r1 = split_and_merge(cv2.merge((r1b, r1g, r1r)))
    r2 = split_and_merge(cv2.merge((r2b, r2g, r2r)))
    r3 = split_and_merge(cv2.merge((r3b, r3g, r3r)))
    
    top = np.hstack((r0, r1))
    bottom = np.hstack((r2, r3))
    img = np.vstack((top, bottom))
        
    return img

image_path = r'C:\Users\soft lab\Desktop\Work3\Image\Lena.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

threshold = int(input("Enter threshold value: "))

#showing Original and segmented image
cv2.imshow("Original RGB Image : ", img)
segmented_image = split_and_merge(img)
cv2.imshow("Segmented Image : ", segmented_image)

#Save output image into Output folder
output_folder = r'C:\Users\soft lab\Desktop\Work3\Output'
output_image_path = os.path.join(output_folder,'Segment.jpg')
cv2.imwrite(output_image_path,segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()