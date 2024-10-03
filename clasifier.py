import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_boosted_classifier(X_train, y_train, save_model=False, model_path="boosted_tree_model.pkl"):

    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # Train the classifier
    classifier.fit(X_train, y_train)
    print("Gradient Boosting Classifier trained successfully.")
    

    if save_model:
        joblib.dump(classifier, model_path)
        print(f"Model saved at: {model_path}")

    return classifier

def predict_with_classifier(classifier, X_test):

    y_pred = classifier.predict(X_test)
    return y_pred

def evaluate_model(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(class_report)

def load_trained_model(model_path):

    classifier = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return classifier

if __name__ == "__main__":

    X_train = np.random.rand(100, 500) 
    y_train = np.random.randint(0, 2, 100) 

    X_test = np.random.rand(20, 500) 
    y_test = np.random.randint(0, 2, 20) 

    classifier = train_boosted_classifier(X_train, y_train, save_model=True, model_path="boosted_tree_model.pkl")


    y_pred = predict_with_classifier(classifier, X_test)

    evaluate_model(y_test, y_pred)
