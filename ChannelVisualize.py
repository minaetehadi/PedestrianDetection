import cv2
from skimage.feature import hog
from skimage import exposure
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt

def extract_luv_and_hog(image):
    # Convert BGR to LUV color space
    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    L, U, V = cv2.split(luv_image)

    hog_features, hog_image = hog(L, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, multichannel=False)


    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return L, U, V, hog_image_rescaled

def compute_gradient_channels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_orientation = np.arctan2(grad_y, grad_x)
    gradient_orientation[gradient_orientation < 0] += np.pi

    # Create orientation channels
    orientation_channels = []
    num_bins = 6
    bin_range = np.linspace(0, np.pi, num_bins + 1)
    for i in range(num_bins):
        lower_bound = bin_range[i]
        upper_bound = bin_range[i + 1]
        channel = ((gradient_orientation >= lower_bound) & (gradient_orientation < upper_bound)).astype(np.float32)
        orientation_channels.append(channel)

    magnitude_channel = gradient_magnitude / np.max(gradient_magnitude)
    return orientation_channels, magnitude_channel

def compute_glcm_features(image):
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = greycomatrix(gray_image, distances, angles, symmetric=True, normed=True)


    contrast = greycoprops(glcm, 'contrast')
    correlation = greycoprops(glcm, 'correlation')
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')

    glcm_features = np.concatenate((contrast.flatten(), correlation.flatten(), energy.flatten(), homogeneity.flatten()))
    return glcm_features, glcm


def compute_optical_flow(prev_image, next_image):
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extract_all_features(image1, image2):
    # Extract LUV and HOG features
    L, U, V, hog_image = extract_luv_and_hog(image1)

    orientation_channels, magnitude_channel = compute_gradient_channels(image1)

    glcm_features, glcm = compute_glcm_features(image1)
    

    flow = compute_optical_flow(image1, image2)
    
    return L, U, V, hog_image, orientation_channels, magnitude_channel, glcm_features, flow

image1_path = '/content/ge1.webp'
image2_path = '/content/ge2.webp'


image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

L, U, V, hog_image, orientation_channels, magnitude_channel, glcm_features, flow = extract_all_features(image1, image2)

plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.imshow(L, cmap='gray')
plt.title('L Channel')

plt.subplot(3, 3, 2)
plt.imshow(U, cmap='gray')
plt.title('U Channel')

plt.subplot(3, 3, 3)
plt.imshow(V, cmap='gray')
plt.title('V Channel')

plt.subplot(3, 3, 4)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Image')

plt.subplot(3, 3, 5)
plt.imshow(magnitude_channel, cmap='gray')
plt.title('Gradient Magnitude')

for i, channel in enumerate(orientation_channels):
    plt.subplot(3, 3, 6 + i)
    plt.imshow(channel, cmap='gray')
    plt.title(f'Orientation Channel {i+1}')

plt.subplot(3, 3, 6 + len(orientation_channels))
plt.imshow(flow[..., 0], cmap='gray')
plt.title('Optical Flow X')

plt.subplot(3, 3, 7 + len(orientation_channels))
plt.imshow(flow[..., 1], cmap='gray')
plt.title('Optical Flow Y')

plt.tight_layout()
plt.show()
