import matplotlib.pyplot as plt
import matplotlib.patches as patches


fig, ax = plt.subplots(figsize=(12, 8))

# Set axis limits and remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis('off')

rects = {
    "init": ((2, 13), "Initialization Phase", "Initialize pheromone levels and create ants"),
    "construct": ((2, 11), "Feature Subset Construction", "Ants select feature subsets"),
    "train": ((2, 9), "Model Training", "Train model with selected features"),
    "attack": ((2, 7), "Adversarial Attack Evaluation", "Apply query-based attack and measure success"),
    "fitness": ((2, 5), "Fitness Calculation", "Compute fitness based on accuracy and robustness"),
    "update": ((2, 3), "Pheromone Update", "Update pheromones based on fitness"),
    "check": ((2, 1), "Convergence Check", "Check stopping criteria")
}

for key, ((x, y), title, text) in rects.items():
    ax.add_patch(patches.Rectangle((x, y), 6, 1.5, edgecolor='black', facecolor='lightgray'))
    ax.text(x + 0.3, y + 1, title, fontsize=12, fontweight='bold')
    ax.text(x + 0.3, y + 0.5, text, fontsize=10)

for i in range(13, 2, -2):
    ax.arrow(5, i, 0, -1, head_width=0.3, head_length=0.3, fc='black', ec='black')

# Output
ax.add_patch(patches.Rectangle((2, -1), 6, 1.5, edgecolor='black', facecolor='lightgray'))
ax.text(2.3, -0.5, "Output", fontsize=12, fontweight='bold')
ax.text(2.3, -1, "Best feature subset with high detection accuracy and robustness", fontsize=10)
ax.arrow(5, 1, 0, -1.5, head_width=0.3, head_length=0.3, fc='black', ec='black')

# Display the plot
plt.show()
