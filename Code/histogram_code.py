import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    return cdf_normalized

def calculate_pdf(hist, num_pixels):
    pdf = hist / float(num_pixels)
    return pdf

def show_histogram(image, title):
    plt.figure()
    plt.hist(image.ravel(), 256, [0,256])
    plt.title(title)
    plt.show()

def show_cdf_pdf(hist, title):
    cdf = calculate_cdf(hist)
    pdf = calculate_pdf(hist, np.prod(hist.shape))

    plt.figure()
    plt.plot(cdf, color='b', label='CDF')
    plt.plot(pdf, color='r', label='PDF')
    plt.title(title)
    plt.legend()
    plt.show()

def show_histogram(image, title):
    plt.figure()
    plt.hist(image.ravel(), 256, [0,256])
    plt.title(title)
    plt.show()

image_path = r'C:\Users\soft lab\Desktop\Work3\Image\Lena.jpg'
input_image = cv2.imread(image_path)

input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
r_channel, g_channel, b_channel = cv2.split(input_image_rgb)

r_channel_eq = cv2.equalizeHist(r_channel)
g_channel_eq = cv2.equalizeHist(g_channel)
b_channel_eq = cv2.equalizeHist(b_channel)

equalized_image_rgb = cv2.merge((r_channel_eq, g_channel_eq, b_channel_eq))
input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

h_channel, s_channel, v_channel = cv2.split(input_image_hsv)
v_equilization = cv2.equalizeHist(v_channel)

equalized_image_hsv = cv2.merge((h_channel, s_channel, v_equilization))
equalized_image_hsv_rgb = cv2.cvtColor(equalized_image_hsv, cv2.COLOR_HSV2RGB)

show_histogram(input_image_rgb, 'Input Image Histogram')

show_histogram(r_channel_eq,'Red channel equilization')
show_histogram(g_channel_eq,'Green channel equilization')
show_histogram(b_channel_eq,'Blue channel equilization')
show_histogram(equalized_image_rgb, 'Equalized RGB Image Histogram')

show_histogram(v_equilization, 'Equalized Value Channel Histogram')
show_histogram(cv2.cvtColor(equalized_image_hsv_rgb, cv2.COLOR_RGB2GRAY), 'Equalized HSV Image Histogram')

input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

input_hist = cv2.calcHist([input_image_gray], [0], None, [256], [0, 256])
output_hist = cv2.calcHist([equalized_image_rgb], [0], None, [256], [0, 256])

show_histogram(input_image_gray, 'Input Image Histogram')
show_histogram(cv2.cvtColor(equalized_image_rgb, cv2.COLOR_RGB2GRAY), 'Equalized RGB Image Histogram')

show_cdf_pdf(input_hist, 'Input Image CDF and PDF')
show_cdf_pdf(output_hist, 'Equalized RGB Image CDF and PDF')

plt.figure()
plt.imshow(input_image_rgb)
plt.title('Input Image')
plt.axis('off')

plt.figure()
plt.imshow(equalized_image_rgb)
plt.title('Equalized RGB Image')
plt.axis('off')

plt.figure()
plt.imshow(cv2.cvtColor(equalized_image_hsv_rgb, cv2.COLOR_RGB2GRAY), cmap='gray')
plt.title('Equalized HSV Image')
plt.axis('off')

plt.show()
