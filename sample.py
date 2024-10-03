import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt

def compute_and_visualize_glcm(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    distances = [1, 2]  # Distance offsets for GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for GLCM
    glcm = greycomatrix(gray_image, distances, angles, symmetric=True, normed=True)

    contrast = greycoprops(glcm, 'contrast')
    correlation = greycoprops(glcm, 'correlation')
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Display GLCM and its properties
    plt.subplot(2, 3, 2)
    plt.imshow(glcm[:, :, 0, 0], cmap='gray', interpolation='nearest')
    plt.title('GLCM (d=1, θ=0)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(glcm[:, :, 0, 1], cmap='gray', interpolation='nearest')
    plt.title('GLCM (d=1, θ=45°)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(glcm[:, :, 1, 0], cmap='gray', interpolation='nearest')
    plt.title('GLCM (d=2, θ=0)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(glcm[:, :, 1, 1], cmap='gray', interpolation='nearest')
    plt.title('GLCM (d=2, θ=45°)')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, f'Contrast: {contrast}\nCorrelation: {correlation}\nEnergy: {energy}\nHomogeneity: {homogeneity}', fontsize=10)
    plt.axis('off')
    plt.title('GLCM Properties')

    plt.tight_layout()
    plt.show()

image_path = '/content/a.jpg'
compute_and_visualize_glcm(image_path)
