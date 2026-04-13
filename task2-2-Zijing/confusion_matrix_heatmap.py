import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Define core plotting function
def plot_and_save_cm(cm, strategy_name, filename):
    labels = ['1', '2', '3', '4', 'NONE']


    # Calculate percentages
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create canvas
    plt.figure(figsize=(10, 8))

    # Draw heatmap
    sns.heatmap(cm_perc, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage (%)'})

    # Set axis labels and title
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix: Participants ({strategy_name})', fontsize=16, fontweight='bold', pad=20)

    # Adjust layout automatically and save the image
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    plt.close()

# Execute Strategy A
cm_a = np.array([
    [27, 0, 2, 8, 15],
    [1, 0, 1, 0, 0],
    [1, 1, 22, 3, 19],
    [40, 1, 38, 239, 119],
    [22, 2, 28, 104, 1349]
])
plot_and_save_cm(cm_a,
                 strategy_name="TF-IDF + LinearSVC",
                 filename="viz_cm_participants_strategy_a.png")

# Execute Strategy B
cm_b = np.array([
    [29, 1, 6, 6, 10],
    [0, 2, 0, 0, 0],
    [5, 1, 23, 3, 14],
    [53, 11, 52, 188, 133],
    [41, 26, 45, 87, 1306]
])
plot_and_save_cm(cm_b,
                 strategy_name="SpaCy Vectors + LinearSVC",
                 filename="viz_cm_participants_strategy_b.png")