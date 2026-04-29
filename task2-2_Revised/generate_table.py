import os
import csv
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")

#  NOTE: Update DATA_ROOT to your local data folder before running
DATA_ROOT = r"D:\Pycharm\TextAnalyticsCW\data"

# Do not modify below
DATA_FILE = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "processed", "ebm_abstracts_full.npz")
CACHE_FILE = os.path.join(DATA_ROOT, "sentence_dataset_cache.npz")
OUTPUT_CSV = os.path.join(DATA_ROOT, "pico_table_ml.csv")

# Load main dataset
print("Loading dataset...")
data = np.load(DATA_FILE, allow_pickle=True)
train_texts = list(data["train_texts"])
test_texts = list(data["test_texts"])
train_masks = {"Pop": data["train_p"], "Int": data["train_i"], "Out": data["train_o"]}
test_masks = {"Pop": data["test_p"], "Int": data["test_i"], "Out": data["test_o"]}
print(f"Train: {len(train_texts)} abstracts | Test: {len(test_texts)} abstracts\n")

# Load SpaCy
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_md")
print("SpaCy model loaded.\n")

# Load or build sentence dataset

if os.path.exists(CACHE_FILE):
    print("Cache found — loading sentence dataset...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    train_sents = list(cache["train_sents"])
    train_sent_labels = {
        "Pop": cache["train_pop"],
        "Int": cache["train_int"],
        "Out": cache["train_out"],
    }
    print(f"Train sentences: {len(train_sents)}\n")

else:
    print("No cache found — building sentence dataset (run pipeline.py first)...")


    def build_sentence_dataset(texts, masks, split_name="Data"):
        all_sents = []
        all_labels = {"Pop": [], "Int": [], "Out": []}
        for idx, text in tqdm(enumerate(texts), total=len(texts),
                              desc=f"Building {split_name} sentences"):
            doc = nlp(text)
            mask_len = len(masks["Pop"][idx])
            for sent in doc.sents:
                if len(sent.text.split()) < 3:
                    continue
                start = min(sent.start, mask_len)
                end = min(sent.end, mask_len)
                all_sents.append(sent.text.strip())
                for key in ["Pop", "Int", "Out"]:
                    mask_slice = list(masks[key][idx][start:end])
                    all_labels[key].append(1 if any(m == 1 for m in mask_slice) else 0)
        return all_sents, {k: np.array(v) for k, v in all_labels.items()}


    train_sents, train_sent_labels = build_sentence_dataset(
        train_texts, train_masks, "Train")

    np.savez_compressed(CACHE_FILE,
                        train_sents=np.array(train_sents, dtype=object),
                        train_pop=train_sent_labels["Pop"],
                        train_int=train_sent_labels["Int"],
                        train_out=train_sent_labels["Out"],
                        )
    print(f"Cache saved to: {CACHE_FILE}\n")


# Fast evaluate_extraction using token index
def evaluate_extraction_fast(sent_text, sent_start, sent_end, gt_mask):
    mask_len = len(gt_mask)
    start = min(sent_start, mask_len)
    end = min(sent_end, mask_len)
    mask_slice = list(gt_mask[start:end])

    if not mask_slice:
        return "None", None

    if any(m == 1 for m in mask_slice):
        return "Hit (Relaxed)", 1.0
    else:
        return "Miss", 0.0


# Train Decomposed model

print("Training TF-IDF + models...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(tqdm(train_sents, desc="  Fitting TF-IDF"))

# Step 1: binary PICO detector
y_bin = np.where(
    (train_sent_labels["Pop"] == 1) |
    (train_sent_labels["Int"] == 1) |
    (train_sent_labels["Out"] == 1), 1, 0)

print("  Training Step 1 (binary detector)...")
clf_step1 = LinearSVC(class_weight="balanced", max_iter=2000)
clf_step1.fit(X_train, y_bin)

# Step 2: per-category classifiers on PICO sentences only
pico_mask = y_bin == 1
clf_step2 = {}
for key in tqdm(["Pop", "Int", "Out"], desc="  Training Step 2 classifiers"):
    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train[pico_mask], train_sent_labels[key][pico_mask])
    clf_step2[key] = clf

print("Models trained.\n")

# Generate table
print("Generating PICO table...")
rows = []

for i, text in enumerate(tqdm(test_texts, desc="Processing test abstracts")):
    doc = nlp(text)

    # Get sentences with their token indices
    sentences = []
    sent_spans = []  # (start, end) token indices
    for sent in doc.sents:
        if len(sent.text.split()) < 3:
            continue
        sentences.append(sent.text.strip())
        sent_spans.append((sent.start, sent.end))

    if not sentences:
        continue

    X_test_doc = tfidf.transform(sentences)
    step1_pred = clf_step1.predict(X_test_doc)
    pico_mask_doc = step1_pred == 1

    row = {"Text": text}

    for key, col in [("Pop", "Pop"), ("Int", "Int"), ("Out", "Out")]:
        extracted_sents = []
        all_statuses = []
        all_precs = []

        if pico_mask_doc.sum() > 0:
            step2_pred = clf_step2[key].predict(X_test_doc[pico_mask_doc])
            pico_indices = np.where(pico_mask_doc)[0]

            for idx, pred in zip(pico_indices, step2_pred):
                if pred == 1:
                    sent_text = sentences[idx]
                    start, end = sent_spans[idx]
                    status, prec = evaluate_extraction_fast(
                        sent_text, start, end, test_masks[key][i])
                    extracted_sents.append(sent_text)
                    all_statuses.append(status)
                    if prec is not None:
                        all_precs.append(prec)

        extracted_text = " ; ".join(extracted_sents) if extracted_sents else "null"

        # Aggregate status across sentences
        if not all_statuses:
            final_status = "None"
            final_prec = None
        elif all(s == "Hit (Relaxed)" for s in all_statuses):
            final_status = "Hit (Relaxed)"
            final_prec = 1.0
        elif all(s == "Miss" for s in all_statuses):
            final_status = "Hallucinated"
            final_prec = 0.0
        else:
            final_status = "Partial Hit"
            final_prec = np.mean(all_precs) if all_precs else 0.0

        row[f"{col}_Extracted"] = extracted_text
        row[f"{col}_Status"] = final_status
        row[f"{col}_Precision"] = final_prec

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
