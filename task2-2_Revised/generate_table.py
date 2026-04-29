import os
import csv
import spacy
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# NOTE: Update DATA_ROOT to your local data folder before running
DATA_ROOT = r"D:\Pycharm\TextAnalyticsCW\data"

# Do not modify below
DATA_DIR = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "documents")
BASE_LABEL_DIR = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "annotations", "aggregated", "hierarchical_labels")
OUTPUT_CSV = os.path.join(DATA_ROOT, "pico_table_ml.csv")

TARGETS = {
    "P": "participants",
    "I": "interventions",
    "O": "outcomes",
}

# Labels that indicate PICO content
PICO_LABELS = {"1", "2", "3", "4", "5", "6", "7"}

# Load SpaCy for noun phrase extraction
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_md")
print("SpaCy model loaded.\n")


#  Data loading
def load_data(short, split="train"):
    suffix = "" if split == "train" else "_test"
    path = os.path.join(DATA_ROOT, f"labeled_data_{short}{suffix}.npz")
    data = np.load(path, allow_pickle=True)
    return list(data["texts"]), data["labels"]


# Load test document IDs
def get_test_doc_ids(target_name):
    test_dir = os.path.join(BASE_LABEL_DIR, target_name, "test", "gold")
    label_files = sorted([f for f in os.listdir(test_dir)
                          if f.endswith(".ann") or f.endswith(".tags")])
    return [f.split(".")[0] for f in label_files]


# Load full abstract text
def get_full_abstract(doc_id):
    token_path = os.path.join(DATA_DIR, doc_id + ".tokens")
    if not os.path.exists(token_path):
        return ""
    with open(token_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f]
    return " ".join(tokens)


# Load sentences per document
def get_doc_sentences(doc_id):
    token_path = os.path.join(DATA_DIR, doc_id + ".tokens")
    if not os.path.exists(token_path):
        return []

    with open(token_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f]

    sentences = []
    sent_tokens = []

    for token in tokens:
        sent_tokens.append(token)
        if token in {".", "?", "!"}:
            if len(sent_tokens) > 4:
                sentences.append(" ".join(sent_tokens))
            sent_tokens = []

    return sentences


#  Noun phrase extraction
def extract_noun_phrases(sentences):
    phrases = []
    for sent in sentences:
        doc = nlp(sent)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            # Filter out very short or uninformative chunks
            if len(phrase.split()) >= 2 and phrase.lower() not in {"the study", "the trial", "the group"}:
                if phrase not in phrases:
                    phrases.append(phrase)
    return phrases


# Train model for one PICO category
def train_decomposed(txt_train, y_train):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(txt_train)

    # Binary detection
    y_bin = np.where(np.isin(y_train, list(PICO_LABELS)), "PICO", "NONE")
    clf1 = LinearSVC(class_weight="balanced", max_iter=2000)
    clf1.fit(X_train, y_bin)

    # Fine-grained classification on PICO sentences only
    pico_mask = np.isin(y_train, list(PICO_LABELS))
    clf2 = LinearSVC(class_weight="balanced", max_iter=2000)
    clf2.fit(X_train[pico_mask], y_train[pico_mask])

    return tfidf, clf1, clf2


def predict_sentences(sentences, tfidf, clf1, clf2):
    if len(sentences) == 0:
        return []

    X = tfidf.transform(sentences)
    step1 = clf1.predict(X)
    labels = np.full(len(sentences), "NONE", dtype=object)
    pico_mask = step1 == "PICO"

    if pico_mask.sum() > 0:
        labels[pico_mask] = clf2.predict(X[pico_mask])

    return list(labels)


# Main

print("Training models...")

models = {}
for short, target_name in TARGETS.items():
    txt_train, y_train = load_data(short, "train")
    tfidf, clf1, clf2 = train_decomposed(txt_train, y_train)
    models[short] = (tfidf, clf1, clf2)
    print(f" [{target_name}] Model trained.")

print("\nGenerating PICO table...")
doc_ids = get_test_doc_ids("participants")

rows = []

for doc_id in doc_ids:
    sentences = get_doc_sentences(doc_id)
    full_abstract = get_full_abstract(doc_id)

    if len(sentences) == 0:
        continue

    row = {"P": [], "I": [], "O": []}

    # Predict and collect PICO sentences per category
    for short in ["P", "I", "O"]:
        tfidf, clf1, clf2 = models[short]
        pred_labels = predict_sentences(sentences, tfidf, clf1, clf2)

        pico_sentences = [sent for sent, lbl in zip(sentences, pred_labels)
                          if lbl != "NONE"]

        # Extract noun phrases from predicted PICO sentences
        row[short] = extract_noun_phrases(pico_sentences)

    rows.append({
        "Text": full_abstract,
        "Pop_Extracted": " ; ".join(row["P"]) if row["P"] else "",
        "Int_Extracted": " ; ".join(row["I"]) if row["I"] else "",
        "Out_Extracted": " ; ".join(row["O"]) if row["O"] else "",
    })

# Save to CSV

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["Text", "Pop_Extracted",
                                           "Int_Extracted", "Out_Extracted"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTable saved to: {OUTPUT_CSV}")
print(f"Total documents: {len(rows)}")

# Preview first 3 rows
print("\nPreview (first 3 rows):")
print(f"{'Doc':<10} {'Pop_Extracted':<45} {'Int_Extracted':<45} {'Out_Extracted':<45}")
print("─" * 148)
for row in rows[:3]:
    p = row["Pop_Extracted"][:42] + "..." if len(row["Pop_Extracted"]) > 45 else row["Pop_Extracted"]
    i = row["Int_Extracted"][:42] + "..." if len(row["Int_Extracted"]) > 45 else row["Int_Extracted"]
    o = row["Out_Extracted"][:42] + "..." if len(row["Out_Extracted"]) > 45 else row["Out_Extracted"]
    text_preview = row["Text"][:10] + "..."
    print(f"{text_preview:<10} {p:<45} {i:<45} {o:<45}")