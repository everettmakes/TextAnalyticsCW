import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
# Use TkAgg backend to ensure the plot opens in a standalone window
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load preprocessed data
print("Loading processed data")
data = np.load("./data/my_processed_data.npz")
vectors = data['vectors']
texts = data['texts']

# Data inspection: display sample sentences
print("\n--- Data Inspection: Sample Sentences ---")
for i in range(3):
    print(f"Sentence {i+1}: {texts[i]}")
    print(f"Vector shape: {vectors[i].shape}\n")
print("-------------------------------------------\n")

# K-Means Clustering
K = 3
print(f"Running K-Means clustering with K={K} ")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(vectors)

# PCA Dimensionality Reduction
print("Performing PCA dimensionality")
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Visualization
print("Generating clustering visualization")
plt.figure(figsize=(10, 8))

# Plot scatter points colored by cluster labels
scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                      c=labels, cmap='viridis', alpha=0.5, s=5)

# Configure plot titles and labels
plt.colorbar(scatter, label='Cluster Category')
plt.title('EBM-NLP Sentence Clustering (K-Means + PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Save the resulting figure
plt.savefig("./data/clustering_result.png")
print("Clustering complete. Image saved.")

# --- 5. Cluster Distribution Statistics / 聚类分布统计 ---
print("Generating distribution statistics...")
unique, counts = np.unique(labels, return_counts=True)
total_samples = len(labels)

# Print text statistics to console
print("\n--- Cluster Distribution Summary ---")
for cluster_id, count in zip(unique, counts):
    percentage = (count / total_samples) * 100
    print(f"Cluster {cluster_id}: {count} sentences ({percentage:.2f}%)")

# Generate bar chart / 生成柱状图
plt.figure(figsize=(8, 6))
colors = ['#440154', '#21918c', '#fde725'] # Match viridis colors / 匹配散点图颜色
plt.bar(unique, counts, color=colors, alpha=0.8)

plt.title('Sentence Distribution per Cluster (N=54,329)')
plt.xlabel('Cluster Category')
plt.ylabel('Number of Sentences')
plt.xticks(unique) # Ensure x-axis shows 0, 1, 2

# Add percentage labels on top of bars
for i, count in enumerate(counts):
    plt.text(i, count + 500, f'{(count/total_samples)*100:.1f}%', ha='center')

plt.savefig("./data/cluster_distribution.png")
print("Distribution chart saved to ./data/cluster_distribution.png")

# Display the plot window
plt.show()