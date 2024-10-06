import cv2
import numpy as np
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

def load_trained_model():
    model = AdaBoostClassifier(n_estimators=50)
    return model

def extract_features(image1, image2):
    luv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2LUV)
    L_channel = luv_image[..., 0]

    hog_features = cv2.HOGDescriptor().compute(L_channel)

    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY),
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_x = flow[..., 0].flatten()
    flow_y = flow[..., 1].flatten()

    features = np.hstack([hog_features.flatten(), flow_x, flow_y])
    return features

def generate_boundary_adversarial_examples(model, X_clean):
    classifier = SklearnClassifier(model=model)  # Wrap your model for ART
    attack = BoundaryAttack(estimator=classifier, targeted=False)
    
    X_adversarial = attack.generate(X=X_clean)
    return X_adversarial

def evaluate_model(model, image_pairs, labels):
    X_clean, y_true = [], labels


    for img1_path, img2_path in image_pairs:
        image1 = cv2.imread(img1_path)
        image2 = cv2.imread(img2_path)

        clean_features = extract_features(image1, image2)
        X_clean.append(clean_features)

    X_clean = np.array(X_clean)

    X_adversarial = generate_boundary_adversarial_examples(model, X_clean)


    y_pred_clean = model.predict(X_clean)
    accuracy_clean = accuracy_score(y_true, y_pred_clean)

    y_pred_adversarial = model.predict(X_adversarial)
    accuracy_adversarial = accuracy_score(y_true, y_pred_adversarial)

    confusion_clean = confusion_matrix(y_true, y_pred_clean)
    confusion_adversarial = confusion_matrix(y_true, y_pred_adversarial)

    print("Evaluation Results:")
    print(f"Accuracy on Clean Data: {accuracy_clean * 100:.2f}%")
    print(f"Confusion Matrix (Clean Data):\n {confusion_clean}")
    
    print(f"Accuracy on Adversarial Examples: {accuracy_adversarial * 100:.2f}%")
    print(f"Confusion Matrix (Adversarial Data):\n {confusion_adversarial}")

if __name__ == "__main__":

    model = load_trained_model()
    

    image_pairs = [("image1.jpg", "image2.jpg"), ("image3.jpg", "image4.jpg")]
    labels = [1, 0]  
    
    evaluate_model(model, image_pairs, labels)
