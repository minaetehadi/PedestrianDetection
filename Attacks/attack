import numpy as np
from art.attacks.evasion import HopSkipJump, BoundaryAttack, SignOptAttack
from art.estimators.classification import SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_boosted_classifier(X_train, y_train, save_model=False, model_path="boosted_tree_model.pkl"):
  
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
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

def apply_blackbox_attack(classifier, X_test, y_test, attack_type="HSJA"):
    
    art_classifier = SklearnClassifier(model=classifier)

    # Choose the attack method based on the type
    if attack_type == "HSJA":
        attack = HopSkipJump(classifier=art_classifier, targeted=False, max_iter=10)
    elif attack_type == "Boundary":
        attack = BoundaryAttack(estimator=art_classifier, targeted=False, max_iter=10)
    elif attack_type == "SignOPT":
        attack = SignOptAttack(classifier=art_classifier, targeted=False, max_iter=10)
    else:
        raise ValueError("Unknown attack type. Please use 'HSJA', 'Boundary', or 'SignOPT'.")

    # Generate adversarial examples
    X_adv = attack.generate(X_test)
    return X_adv


if __name__ == "__main__":
    X_train = np.random.rand(100, 500)  # 100 training samples with 500 features each
    y_train = np.random.randint(0, 2, 100)  # Binary labels for training

    X_test = np.random.rand(20, 500)  # 20 test samples with 500 features each
    y_test = np.random.randint(0, 2, 20)  # Binary labels for testing

    classifier = train_boosted_classifier(X_train, y_train, save_model=True, model_path="boosted_tree_model.pkl")
    print("Model trained successfully.")

    evaluate_model(y_test, predict_with_classifier(classifier, X_test))

    attack_type = "HSJA"  # Choose between 'HSJA', 'Boundary', 'SignOPT'
    print(f"Applying {attack_type} attack...")
    X_adv = apply_blackbox_attack(classifier, X_test, y_test, attack_type)

    print("Evaluating on adversarial examples...")
    evaluate_model(y_test, predict_with_classifier(classifier, X_adv))
