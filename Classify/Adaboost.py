import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def extract_features(image1, image2):

    acf_features = extract_acf(image1)
    lbp_features = extract_lbpcm(image1)
    flow_x, flow_y = extract_optical_flow(image1, image2)

    lbp_features = np.expand_dims(lbp_features, axis=2)
    flow_x = np.expand_dims(flow_x, axis=2)
    flow_y = np.expand_dims(flow_y, axis=2)

    combined_features = np.concatenate([acf_features, lbp_features, flow_x, flow_y], axis=2)

    return combined_features.flatten()

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            predictions = stump.predict(X)
            misclassified = (predictions != y).astype(int)
            err = np.dot(w, misclassified) / np.sum(w)
            alpha = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * y * predictions)
            w = w / np.sum(w)
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            predictions = model.predict(X)
            final_predictions += alpha * predictions
        return np.sign(final_predictions)

image_pairs = [("image1.jpg", "image2.jpg"), ("image3.jpg", "image4.jpg")] 
labels = [1, -1, 1, -1, ...] 

# Extract features for each image pair
X = []
y = labels

for img1_path, img2_path in image_pairs:
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    features = extract_features(image1, image2)
    X.append(features)

X = np.array(X)  # Convert list of features to NumPy array
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X_train, y_train)

y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

