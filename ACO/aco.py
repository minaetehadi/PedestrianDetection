import numpy as np
def initialize_pheromones(num_features):

    return np.full(num_features, 0.1)

def evaluate_feature_subset(features, classifier, X, y):

    classifier.fit(X[:, features], y)
    return classifier.score(X[:, features], y)

def update_pheromones(pheromones, features, evaluation, evaporation_rate):

    pheromones[features] += evaluation * (1 - evaporation_rate)
    pheromones *= evaporation_rate  

def select_features(pheromones, num_selected_features):

    probabilities = pheromones / np.sum(pheromones)
    selected_features = np.random.choice(len(pheromones), num_selected_features, p=probabilities, replace=False)
    return selected_features

def ant_colony_optimization(X, y, classifier, num_ants=50, num_features=10, evaporation_rate=0.2, iterations=100):
    num_total_features = X.shape[1]
    pheromones = initialize_pheromones(num_total_features)
    
    for _ in range(iterations):
        for _ in range(num_ants):
            selected_features = select_features(pheromones, num_features)
            
            evaluation = evaluate_feature_subset(selected_features, classifier, X, y)

            update_pheromones(pheromones, selected_features, evaluation, evaporation_rate)
    
    best_features = select_features(pheromones, num_features)
    return best_features
