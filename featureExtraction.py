import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def extract_acf(image):
    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    
    hog_features, hog_image = hog(luv_image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, multichannel=True)
    
    gradient_x = cv2.Sobel(luv_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luv_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    acf_features = np.concatenate([luv_image.flatten(), hog_features, gradient_magnitude.flatten()])
    
    return acf_features


def extract_lbpcm(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    
    co_matrix = np.histogram(lbp.ravel(), bins=np.arange(0, 256))
    
    return co_matrix[0]

def extract_optical_flow(image1, image2):
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    
    return flow_x, flow_y
