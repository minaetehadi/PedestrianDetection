import os
import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from art.attacks.evasion import HopSkipJump, BoundaryAttack
from art.estimators.classification import SklearnClassifier

def load_inria_data(dataset_path, image_size=(64, 128)):
    """Loads ETH dataset."""
    X, y = [], []
    
    pos_dir = os.path.join(dataset_path, "pos")
    for file_name in os.listdir(pos_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            img_path = os.path.join(pos_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)  # Resize to uniform size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            X.append(img.flatten())  # Flatten the image to a feature vector
            y.append(1)  

    neg_dir = os.path.join(dataset_path, "neg")
    for file_name in os.listdir(neg_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            img_path = os.path.join(neg_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)  # Resize to uniform size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            X.append(img.flatten())  # Flatten the image to a feature vector
            y.append(0)  
    
    # Convert to numpy arrays and normalize pixel values
    X = np.array(X).astype('float32') / 255.0
    y = np.array(y)
    
    return X, y

def initialize_pheromone(n_features):

    return np.ones((n_features, n_features))

def evaluate_subset(selected_features, X_train, y_train, X_test, y_test, eval_func):

    selected_X_train = X_train[:, selected_features == 1]
    selected_X_test = X_test[:, selected_features == 1]

    model = GradientBoostingClassifier()
    model.fit(selected_X_train, y_train)

    return eval_func(model, selected_X_test, y_test)

def update_pheromone(Tk, delta_tau, rho):

    return (1 - rho) * Tk + rho * delta_tau

def step_function(Tk, alpha, beta, current_feature, selected_features, inverse_corr):
    """Select the next feature based on pheromone and attractiveness."""
    n_features = len(Tk)
    max_attractiveness = -1
    next_feature = None

    for feature in range(n_features):
        if selected_features[feature] == 0:  # Feature not yet selected
            attractiveness = (Tk[current_feature, feature] ** alpha) * (inverse_corr[current_feature, feature] ** beta)
            if attractiveness > max_attractiveness:
                max_attractiveness = attractiveness
                next_feature = feature

    selected_features[next_feature] = 1 
    return next_feature, selected_features

def adversarial_robustness(model, X_test, y_test, attack_type="hsja"):

    classifier = SklearnClassifier(model=model)

    if attack_type == "hsja":
        attack = HopSkipJump(classifier=classifier)
    elif attack_type == "boundary":
        attack = BoundaryAttack(classifier=classifier)

    X_adv = attack.generate(X=X_test)

    y_pred_adv = model.predict(X_adv)
    return accuracy_score(y_test, y_pred_adv)  # Return success rate on adversarial examples

def ant_colony_optimization(X_train, y_train, X_test, y_test, n_features, m, delta, rho, alpha, beta, max_iter, attack_type):

    Tk = initialize_pheromone(n_features) 
    inverse_corr = np.random.rand(n_features, n_features)
    best_eval = -1
    best_features = np.zeros(n_features)

    for iteration in range(max_iter):
        delta_tau = np.zeros_like(Tk)

        for ant in range(m):
            selected_features = np.zeros(n_features)
            current_feature = np.random.randint(n_features)
            selected_features[current_feature] = 1

            while np.sum(selected_features) < delta * n_features:
                next_feature, selected_features = step_function(Tk, alpha, beta, current_feature, selected_features, inverse_corr)
                delta_tau[current_feature, next_feature] += 1
                current_feature = next_feature

            evaluation = evaluate_subset(selected_features, X_train, y_train, X_test, y_test, 
                                         lambda model, X_test, y_test: adversarial_robustness(model, X_test, y_test, attack_type))
            if evaluation > best_eval:
                best_eval = evaluation
                best_features = np.copy(selected_features)

        Tk = update_pheromone(Tk, delta_tau, rho)

    return best_features, best_eval

dataset_path = "path_to_dataset"
X, y = load_ETH_data(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


n_features = X_train.shape[1]  # Number of features
m = 50  # Number of ants
delta = 0.25  
rho = 0.2  # Evaporation rate
alpha = 0.9  # Importance of pheromone
beta = 0.4  # Importance of feature correlation
max_iter = 100  # Maximum ACO iterations
attack_type = "hsja" 


best_features, best_eval = ant_colony_optimization(X_train, y_train, X_test, y_test, n_features, m, delta, rho, alpha, beta, max_iter, attack_type)

print("Best feature subset:", best_features)
print("Best evaluation (robustness):", best_eval)
