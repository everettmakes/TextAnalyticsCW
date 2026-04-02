import os
import spacy
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define data paths
DATA_DIR = Path("./data/ebm_nlp_2_00")
SAVE_PATH = Path("./data/my_processed_data.npz")

# Load the Spacy English model
print("Loading Spacy model...")
nlp = spacy.load("en_core_web_md")


def vectorize_my_data(num_docs):
    """
    Extract sentences from documents and convert them to word vectors.
    """
    # Get the list of document files
    doc_files = list((DATA_DIR / "documents").glob("*.tokens"))[:num_docs]

    all_vectors = []
    all_texts = []

    print(f"Processing {len(doc_files)} documents... / ")
    for file_path in tqdm(doc_files):
        with open(file_path, "r", encoding="utf-8") as f:
            # Reconstruct full text from tokens
            text = " ".join([line.strip() for line in f])
            doc = nlp(text)

            # Split into sentences and vectorize
            for sent in doc.sents:
                # Filter out very short sentences (<= 4 words)
                if len(sent.text.split()) > 4:
                    all_vectors.append(sent.vector)
                    all_texts.append(sent.text)

    return np.array(all_vectors), all_texts


# Execute the vectorization process
vectors, texts = vectorize_my_data(None)

# Save the results to a compressed numpy file
np.savez_compressed(SAVE_PATH, vectors=vectors, texts=texts)

print(f"Data successfully saved to : {SAVE_PATH}")
print(f"Total sentence vectors extracted : {len(vectors)}")