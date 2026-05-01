import os
import csv
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import warnings
warnings.filterwarnings("ignore")

#  NOTE: Update DATA_ROOT to your local data folder before running
DATA_ROOT = r"D:\Pycharm\TextAnalyticsCW\data"

# Do not modify below
DATA_FILE = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "processed", "ebm_abstracts_full.npz")
CACHE_FILE = os.path.join(DATA_ROOT, "token_dataset_cache.npz")
OUTPUT_CSV = os.path.join(DATA_ROOT, "pico_table_ml.csv")

CONTEXT_WINDOW = 2


# Load data

print("Loading dataset...")
data = np.load(DATA_FILE, allow_pickle=True)
train_texts = list(data["train_texts"])
test_texts = list(data["test_texts"])
train_masks = {"Pop": data["train_p"], "Int": data["train_i"], "Out": data["train_o"]}
test_masks = {"Pop": data["test_p"],  "Int": data["test_i"],  "Out": data["test_o"]}
print(f"Train: {len(train_texts)} abstracts | Test: {len(test_texts)} abstracts\n")


# Token feature extraction
def token_features(tokens, i, window=CONTEXT_WINDOW):
    features = {}
    token = tokens[i]

    features["token"] = token.lower()
    features["is_upper"] = int(token.isupper())
    features["is_title"] = int(token.istitle())
    features["is_digit"] = int(token.isdigit())
    features["token_length"] = len(token)

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        j = i + offset
        features[f"context_{offset}"] = tokens[j].lower() if 0 <= j < len(tokens) else "<PAD>"

    return features


def build_token_dataset(texts, masks, split_name="Data"):

    all_features = []
    all_labels = {"Pop": [], "Int": [], "Out": []}
    doc_lengths = []

    for idx, text in tqdm(enumerate(texts), total=len(texts),
                          desc=f"Building {split_name} token features"):
        tokens = text.split()
        n_tokens = len(tokens)
        doc_lengths.append(n_tokens)

        for i in range(n_tokens):
            all_features.append(token_features(tokens, i))

        for key in ["Pop", "Int", "Out"]:
            mask = list(masks[key][idx])
            if len(mask) >= n_tokens:
                all_labels[key].extend(mask[:n_tokens])
            else:
                all_labels[key].extend(mask + [0] * (n_tokens - len(mask)))

    return (all_features,
            {k: np.array(v, dtype=int) for k, v in all_labels.items()},
            doc_lengths)


#  Load or build token dataset
if os.path.exists(CACHE_FILE):
    print("Cache found — loading token dataset...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    train_features = list(cache["train_features"])
    test_features = list(cache["test_features"])
    train_doc_lengths = list(cache["train_doc_lengths"])
    test_doc_lengths = list(cache["test_doc_lengths"])
    train_labels = {
        "Pop": cache["train_pop"],
        "Int": cache["train_int"],
        "Out": cache["train_out"],
    }
    print(f"Train tokens: {len(train_features)} | Test tokens: {len(test_features)}\n")

else:
    print("No cache found — building token dataset (run pipeline.py first)...")
    train_features, train_labels, train_doc_lengths = \
        build_token_dataset(train_texts, train_masks, "Train")
    test_features, _, test_doc_lengths = \
        build_token_dataset(test_texts, test_masks, "Test")

    np.savez_compressed(CACHE_FILE,
        train_features=np.array(train_features, dtype=object),
        test_features=np.array(test_features,   dtype=object),
        train_doc_lengths=np.array(train_doc_lengths),
        test_doc_lengths=np.array(test_doc_lengths),
        train_pop=train_labels["Pop"],
        train_int=train_labels["Int"],
        train_out=train_labels["Out"],
    )
    print(f"Cache saved to: {CACHE_FILE}\n")

# Vectorize features
print("Vectorizing features...")
vec = DictVectorizer(sparse=True)
X_train = vec.fit_transform(train_features)
print("Done.\n")


#  Train Decomposed model
print("Training models...")

# Step 1: binary PICO detector
y_bin = np.where(
    (train_labels["Pop"] == 1) |
    (train_labels["Int"] == 1) |
    (train_labels["Out"] == 1), 1, 0)

print("  Training Step 1 (binary detector)...")
clf_step1 = LinearSVC(class_weight="balanced", max_iter=2000)
clf_step1.fit(X_train, y_bin)

# Step 2: per-category on PICO tokens only
pico_mask = y_bin == 1
clf_step2 = {}
for key in tqdm(["Pop", "Int", "Out"], desc="  Training Step 2 classifiers"):
    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train[pico_mask], train_labels[key][pico_mask])
    clf_step2[key] = clf

print("Models trained.\n")


# Evaluation
def evaluate_extraction(original_text, extracted_text, gt_mask, threshold=0.6):
    if not extracted_text or str(extracted_text).lower() in ("null", "none"):
        return "None", None

    items = [item.strip() for item in str(extracted_text).split(";")]
    orig_words = original_text.split()
    item_scores = []

    for item in items:
        if not item:
            continue
        ext_words = item.split()
        window = len(ext_words)
        best_ratio, best_start = 0, 0

        for i in range(len(orig_words) - window + 1):
            window_text = " ".join(orig_words[i:i + window])
            ratio = SequenceMatcher(None, item.lower(), window_text.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio < threshold:
            item_scores.append(0.0)
        else:
            mask_slice = list(gt_mask[best_start: best_start + window])
            if any(label == 1 for label in mask_slice):
                item_scores.append(1.0)
            else:
                item_scores.append(0.0)

    if not item_scores:
        return "None", None

    avg_precision = sum(item_scores) / len(item_scores)
    if avg_precision == 1.0:
        return "Hit (Relaxed)", 1.0
    elif avg_precision == 0.0:
        return "Hallucinated", 0.0
    else:
        return "Partial Hit", avg_precision


# Generate table
print("Generating PICO table...")
rows = []
tok_idx = 0

for i, text in enumerate(tqdm(test_texts, desc="Processing test abstracts")):
    tokens = text.split()
    n_tokens = test_doc_lengths[i]

    # Get token features for this document
    doc_features = test_features[tok_idx: tok_idx + n_tokens]
    tok_idx += n_tokens

    if not doc_features:
        continue

    X_doc = vec.transform(doc_features)
    step1_pred = clf_step1.predict(X_doc)
    pico_mask_doc = step1_pred == 1

    row = {"Text": text}

    for key, col in [("Pop", "Pop"), ("Int", "Int"), ("Out", "Out")]:
        # Step 2 on predicted PICO tokens
        final_pred = np.zeros(n_tokens, dtype=int)
        if pico_mask_doc.sum() > 0:
            step2_pred = clf_step2[key].predict(X_doc[pico_mask_doc])
            final_pred[pico_mask_doc] = step2_pred

        # Merge contiguous PICO tokens into phrases
        phrases = []
        current = []
        for token, pred in zip(tokens[:n_tokens], final_pred):
            if pred == 1:
                current.append(token)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))

        extracted_text = " ; ".join(phrases) if phrases else "null"
        status, prec = evaluate_extraction(text, extracted_text, test_masks[key][i])

        row[f"{col}_Extracted"] = extracted_text
        row[f"{col}_Status"] = status
        row[f"{col}_Precision"] = prec

    rows.append(row)


# Save CSV
fieldnames = ["Text",
              "Pop_Extracted", "Pop_Status", "Pop_Precision",
              "Int_Extracted", "Int_Status", "Int_Precision",
              "Out_Extracted", "Out_Status", "Out_Precision"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTable saved to: {OUTPUT_CSV}")
print(f"Total documents: {len(rows)}")

#  Summary
print("\nSummary:")
print(f"  {'Category':<8} {'Precision':>10} {'Hit':>6} {'Partial':>8} {'Miss':>6} {'Halluc':>8}")
print("  " + "─" * 50)

for col in ["Pop", "Int", "Out"]:
    precs = [r[f"{col}_Precision"] for r in rows if r[f"{col}_Precision"] is not None]
    statuses = [r[f"{col}_Status"] for r in rows]
    mean_p = np.mean(precs) if precs else 0.0
    hits = statuses.count("Hit (Relaxed)")
    partial = statuses.count("Partial Hit")
    misses = statuses.count("Miss")
    halluc = statuses.count("Hallucinated")
    print(f"  {col:<8} {mean_p:>10.3f} {hits:>6} {partial:>8} {misses:>6} {halluc:>8}")