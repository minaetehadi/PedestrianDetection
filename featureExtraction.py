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


    acf_features = np.concatenate([luv_image, gradient_x, gradient_y], axis=2)
    
    return acf_features

def extract_lbpcm(image):
 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    
    return lbp

def extract_optical_flow(image1, image2):
    # Convert images to grayscale for optical flow calculation
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_image1, gray_image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    
    return flow_x, flow_y

def combine_features(image1, image2):
   
    acf_features = extract_acf(image1)
    

    lbp_features = extract_lbpcm(image1)
    lbp_features = np.expand_dims(lbp_features, axis=2)  # Add a channel dimension to LBP

    flow_x, flow_y = extract_optical_flow(image1, image2)
    flow_x = np.expand_dims(flow_x, axis=2)  # Add a channel dimension
    flow_y = np.expand_dims(flow_y, axis=2)  # Add a channel dimension
    
    # Concatenate all features to form a 13-channel feature map
    combined_features = np.concatenate([acf_features, lbp_features, flow_x, flow_y], axis=2)
    
    return combined_features

image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

combined_feature_map = combine_features(image1, image2)
print(f'Combined feature map shape: {combined_feature_map.shape}')  #  HxWx13
