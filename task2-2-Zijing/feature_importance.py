import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# Configure global parameters
TARGETS = ['P', 'I', 'O']
TITLES = {'P': 'Participants', 'I': 'Interventions', 'O': 'Outcomes'}

# Create a 1-row, 3-column large canvas
fig, axes = plt.subplots(1, 3, figsize=(20, 8))


# Loop through the three categories and train the model
for i, target in enumerate(TARGETS):
    train_path = f"./data/labeled_data_{target}.npz"


    #  ensure the file exists
    if not os.path.exists(train_path):
        print(f"Warning: Cannot find {train_path}, skipping {target}.")
        continue

    # Load data and extract TF-IDF features
    data_train = np.load(train_path)
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train = tfidf.fit_transform(data_train['texts'])

    # Train Linear SVM with balanced class weights
    clf = LinearSVC(class_weight='balanced')
    clf.fit(X_train, data_train['labels'])


    # Extract feature weights and select Top words
    class_index = 0
    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = clf.coef_[class_index]

    # Select the top 12 words with the strongest positive and negative contributions
    top_pos_indices = np.argsort(coefficients)[-12:]
    top_neg_indices = np.argsort(coefficients)[:12]
    combined_indices = np.concatenate((top_neg_indices, top_pos_indices))

    #  Plot on the corresponding subplot
    ax = axes[i]

    # Red for negative features, green for positive features
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coefficients[combined_indices]]
    bars = ax.barh(np.arange(len(combined_indices)), coefficients[combined_indices], color=colors, alpha=0.85)

    # Set subplot details
    ax.set_yticks(np.arange(len(combined_indices)))
    ax.set_yticklabels(feature_names[combined_indices], fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)

    ax.set_title(f'{TITLES[target]}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)


# Set the main title
fig.suptitle('Top Predictive Features Across PICO Entities (TF-IDF + LinearSVC Baseline)', fontsize=22,
             fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Export as image
plt.savefig('viz_features_three_panel.png', dpi=300, bbox_inches='tight')
plt.show()

plt.show()