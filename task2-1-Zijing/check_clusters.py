import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import re
# Load data and clustering results
# Load original preprocessed data
data = np.load("./data/my_processed_data.npz")
texts = data['texts']
vectors = data['vectors']

# Re-run K-Means to align labels for inspection
print("Aligning cluster labels")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(vectors)

# Sample and inspect cluster content
print("\n" + "=" * 50)
print("Cluster Content Inspection")
print("=" * 50)

for cluster_id in range(3):
    print(f"\nCluster {cluster_id} Samples")

    # Identify indices belonging to the current cluster
    indices = np.where(labels == cluster_id)[0]

    # Randomly select 5 samples for review
    sample_indices = np.random.choice(indices, min(5, len(indices)), replace=False)

    for i, idx in enumerate(sample_indices):
        print(f"  {i + 1}. {texts[idx]}")
    print("-" * 30)

    # Simple Keyword Analysis
    print("\n" + "=" * 50)
    print("Top Keywords per Cluster (Excluding common stop words)")
    print("=" * 50)

    # Basic stop words list
    stop_words = {'the', 'and', 'was', 'with', 'for', 'were', 'patients', 'study', 'group', 'from', 'that'}

    for cluster_id in range(3):
        print(f"\nTop terms in Cluster {cluster_id}:")
        indices = np.where(labels == cluster_id)[0]

        # Combine all text in this cluster
        all_text = " ".join([texts[idx].lower() for idx in indices])
        # Simple tokenization
        words = re.findall(r'\b\w{3,}\b', all_text)  # Only words with 3+ letters

        # Filter and count
        meaningful_words = [w for w in words if w not in stop_words]
        most_common = Counter(meaningful_words).most_common(10)

        # Format output: word (count)
        keywords_str = ", ".join([f"{word} ({count})" for word, count in most_common])
        print(f"  {keywords_str}")