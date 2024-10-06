import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def compute_hog(image, cell_size=8, block_size=2, nbins=9):
    hog = cv2.HOGDescriptor((image.shape[1], image.shape[0]), (block_size*cell_size, block_size*cell_size), 
                            (cell_size, cell_size), (cell_size, cell_size), nbins)
    return hog.compute(image)


def compute_optical_flow(prev_frame, curr_frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang

def compute_lbp(image, radius=1, n_points=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points * radius, radius, method="uniform")
    return lbp

def extract_acf_features(image, prev_image=None):
    features = []
    

    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    L, U, V = cv2.split(luv_image)
    features.extend([L.flatten(), U.flatten(), V.flatten()])
    

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = compute_hog(gray_image)
    features.append(hog_features.flatten())
    

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    features.append(grad_magnitude.flatten())

    lbp_features = compute_lbp(image)
    features.append(lbp_features.flatten())
    

    if prev_image is not None:
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mag, ang = compute_optical_flow(prev_gray, curr_gray)
        features.extend([mag.flatten(), ang.flatten()])
    

    return np.concatenate(features)

def feature_selection_with_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def load_and_process_data(video_file, labels):
    capture = cv2.VideoCapture(video_file)
    prev_frame = None
    X = []
    y = []


    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        features = extract_acf_features(frame, prev_frame)
        X.append(features)

        y.append(labels.pop(0)) 
        

        prev_frame = frame
    
    capture.release()
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def main(video_file, labels):
    # Load and process the data
    X, y = load_and_process_data(video_file, labels)


    X_reduced = feature_selection_with_pca(X, n_components=100)


    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)


    classifier = GradientBoostingClassifier()
    classifier.fit(X_train, y_train)


    y_pred = classifier.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return classifier


if __name__ == "__main__":

    labels = [1, 0, 1, 0, 1] 
    

    classifier = main('input_video.mp4', labels)
