import os
import csv
import numpy as np
import spacy
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT / "data"

# Do not modify below
DATA_FILE = DATA_ROOT / "ebm_nlp_2_00" / "processed" / "ebm_abstracts_full.npz"
CACHE_FILE = SCRIPT_DIR / "sentence_dataset_cache.npz"
E2E_OUTPUT_CSV = SCRIPT_DIR / "ml_end_to_end_predictions.csv"
DECOMPOSED_OUTPUT_CSV = SCRIPT_DIR / "ml_decomposed_predictions.csv"

print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_md")
print("SpaCy model loaded.\n")

# Load data
print("Loading dataset...")
data = np.load(DATA_FILE, allow_pickle=True)
train_texts = list(data["train_texts"])
test_texts = list(data["test_texts"])
train_masks = {"Pop": data["train_p"], "Int": data["train_i"], "Out": data["train_o"]}
test_masks = {"Pop": data["test_p"], "Int": data["test_i"], "Out": data["test_o"]}
test_ids = list(data["test_ids"]) if "test_ids" in data.files else [
    f"test_{i}" for i in range(len(test_texts))
]
print(f"Train abstracts: {len(train_texts)} | Test abstracts: {len(test_texts)}\n")

if not train_texts or not test_texts:
    raise ValueError(
        f"{DATA_FILE} contains no train/test abstracts. "
        "Run task2-ml/prepare_data.py after checking the raw EBM-NLP data path."
    )


# Sentence segmentation
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


# Load or build sentence dataset
def build_and_save_sentence_cache():
    print("Building sentence-level datasets...")
    train_sents, train_sent_labels = build_sentence_dataset(train_texts, train_masks, "Train")
    test_sents, test_sent_labels = build_sentence_dataset(test_texts, test_masks, "Test")
    print(f"Train sentences: {len(train_sents)} | Test sentences: {len(test_sents)}\n")

    if not train_sents or not test_sents:
        raise ValueError(
            "Sentence dataset is empty. Check that ebm_abstracts_full.npz was "
            "created from the raw EBM-NLP documents/annotations folders."
        )

    np.savez_compressed(CACHE_FILE,
                        train_sents=np.array(train_sents, dtype=object),
                        test_sents=np.array(test_sents, dtype=object),
                        train_pop=train_sent_labels["Pop"],
                        train_int=train_sent_labels["Int"],
                        train_out=train_sent_labels["Out"],
                        test_pop=test_sent_labels["Pop"],
                        test_int=test_sent_labels["Int"],
                        test_out=test_sent_labels["Out"],
                        )
    print(f"Cache saved to: {CACHE_FILE}\n")
    return train_sents, test_sents, train_sent_labels, test_sent_labels


if os.path.exists(CACHE_FILE):
    print("Cache found — loading sentence dataset...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    train_sents = list(cache["train_sents"])
    test_sents = list(cache["test_sents"])
    train_sent_labels = {
        "Pop": cache["train_pop"],
        "Int": cache["train_int"],
        "Out": cache["train_out"],
    }
    test_sent_labels = {
        "Pop": cache["test_pop"],
        "Int": cache["test_int"],
        "Out": cache["test_out"],
    }
    print(f"Train sentences: {len(train_sents)} | Test sentences: {len(test_sents)}\n")

    if not train_sents or not test_sents:
        print("Cached sentence dataset is empty; rebuilding it.")
        train_sents, test_sents, train_sent_labels, test_sent_labels = build_and_save_sentence_cache()

else:
    train_sents, test_sents, train_sent_labels, test_sent_labels = build_and_save_sentence_cache()


#  Evaluation
def extract_text_from_mask(text, mask):
    words = text.split()
    spans = []
    current = []
    for word, label in zip(words, mask):
        if label == 1:
            current.append(word)
        else:
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return " ; ".join(spans) if spans else "null"


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
                item_scores.append(1.0)  # Hit
            else:
                item_scores.append(0.0)  # Miss

    if not item_scores:
        return "None", None

    avg_precision = sum(item_scores) / len(item_scores)
    if avg_precision == 1.0:
        return "Hit (Relaxed)", 1.0
    elif avg_precision == 0.0:
        return "Hallucinated", 0.0
    else:
        return "Partial Hit", avg_precision


def score_predictions(test_texts, test_masks, predicted_spans, category):
    statuses = []
    precisions = []

    for i, (text, span) in enumerate(zip(test_texts, predicted_spans)):
        status, prec = evaluate_extraction(text, span, test_masks[category][i])
        statuses.append(status)
        precisions.append(prec if prec is not None else 0.0)

    hits = statuses.count("Hit (Relaxed)")
    partial = statuses.count("Partial Hit")
    misses = statuses.count("Miss")
    halluc = statuses.count("Hallucinated")
    mean_prec = np.mean(precisions)

    return {"Hit": hits, "Partial": partial, "Miss": misses,
            "Hallucinated": halluc, "Mean_Precision": mean_prec}


def build_prediction_rows(method, predicted_spans):
    rows = []
    for i, text in enumerate(test_texts):
        rows.append({
            "method": method,
            "doc_id": test_ids[i],
            "text": text,
            "Pop_gold": extract_text_from_mask(text, test_masks["Pop"][i]),
            "Pop_pred": predicted_spans["Pop"][i],
            "Int_gold": extract_text_from_mask(text, test_masks["Int"][i]),
            "Int_pred": predicted_spans["Int"][i],
            "Out_gold": extract_text_from_mask(text, test_masks["Out"][i]),
            "Out_pred": predicted_spans["Out"][i],
        })
    return rows


def save_prediction_table(rows, output_path):
    fieldnames = [
        "method", "doc_id", "text",
        "Pop_gold", "Pop_pred",
        "Int_gold", "Int_pred",
        "Out_gold", "Out_pred",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Unified prediction table saved to: {output_path}")


# TF-IDF feature extraction
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(train_sents)
X_test = tfidf.transform(test_sents)


# Extract spans from predicted PICO sentences

def spans_from_sentence_preds(pred_labels, test_texts, test_sents_per_doc):
    spans = []
    sent_idx = 0

    for i, text in enumerate(test_texts):
        n_sents = test_sents_per_doc[i]
        doc_preds = pred_labels[sent_idx: sent_idx + n_sents]
        doc_sents = test_sents[sent_idx: sent_idx + n_sents]
        sent_idx += n_sents

        pico_sents = [s for s, p in zip(doc_sents, doc_preds) if p == 1]
        spans.append(" ; ".join(pico_sents) if pico_sents else "null")

    return spans


# Count sentences per test document
test_sents_per_doc = [sum(1 for s in nlp(t).sents if len(s.text.split()) >= 3)
                      for t in test_texts]

# Model 1: End-to-End
print("=" * 60)
print("Model 1 — End-to-End")
print("=" * 60)
print("Hypothesis: single classifier handles all three PICO categories")
print("simultaneously, may conflate categories.\n")

e2e_spans = {}

for category in ["Pop", "Int", "Out"]:
    y_train = train_sent_labels[category]
    y_test = test_sent_labels[category]

    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n  [{category}] Sentence-level Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-PICO", "PICO"],
                                zero_division=0))

    spans = spans_from_sentence_preds(y_pred, test_texts, test_sents_per_doc)
    scores = score_predictions(test_texts, test_masks, spans, category)
    e2e_spans[category] = spans

    print(f"  [{category}] Span Extraction Scores (vs word-level gt mask):")
    print(f"    Hit: {scores['Hit']}  Partial: {scores['Partial']}  "
          f"Miss: {scores['Miss']}  Hallucinated: {scores['Hallucinated']}")
    print(f"    Mean Precision: {scores['Mean_Precision']:.3f}")

save_prediction_table(
    build_prediction_rows("ml_end_to_end", e2e_spans),
    E2E_OUTPUT_CSV,
)


# Model 2: Decomposed
print("\n" + "=" * 60)
print("Model 2 — Decomposed")
print("=" * 60)
print("Hypothesis: Step 1 filters non-PICO sentences, Step 2 classifies")
print("category. Reduces noise but error propagates from Step 1.\n")

# Step 1: binary
y_train_bin = np.where(
    (train_sent_labels["Pop"] == 1) |
    (train_sent_labels["Int"] == 1) |
    (train_sent_labels["Out"] == 1), 1, 0)

y_test_bin = np.where(
    (test_sent_labels["Pop"] == 1) |
    (test_sent_labels["Int"] == 1) |
    (test_sent_labels["Out"] == 1), 1, 0)

clf_step1 = LinearSVC(class_weight="balanced", max_iter=2000)
clf_step1.fit(X_train, y_train_bin)
step1_pred = clf_step1.predict(X_test)

print("  [Step 1 — Binary: PICO vs Non-PICO]")
print(classification_report(y_test_bin, step1_pred,
                            target_names=["Non-PICO", "PICO"],
                            zero_division=0))

# Step 2: per-category classifier trained only on PICO sentences
pico_train_mask = y_train_bin == 1
pico_test_mask = step1_pred == 1

X_train_pico = X_train[pico_train_mask]
X_test_pico = X_test[pico_test_mask]

dec_spans = {}

for category in ["Pop", "Int", "Out"]:
    y_train_cat = train_sent_labels[category][pico_train_mask]

    clf_step2 = LinearSVC(class_weight="balanced", max_iter=2000)
    clf_step2.fit(X_train_pico, y_train_cat)
    step2_pred = clf_step2.predict(X_test_pico)

    # Combine
    final_pred = np.zeros(len(test_sents), dtype=int)
    final_pred[pico_test_mask] = step2_pred

    print(f"\n  [{category}] Step 2 — Fine-grained Classification Report:")
    print(classification_report(test_sent_labels[category], final_pred,
                                target_names=["Non-PICO", "PICO"],
                                zero_division=0))

    spans = spans_from_sentence_preds(final_pred, test_texts, test_sents_per_doc)
    scores = score_predictions(test_texts, test_masks, spans, category)
    dec_spans[category] = spans

    print(f"  [{category}] Span Extraction Scores (vs word-level gt mask):")
    print(f"    Hit: {scores['Hit']}  Partial: {scores['Partial']}  "
          f"Miss: {scores['Miss']}  Hallucinated: {scores['Hallucinated']}")
    print(f"    Mean Precision: {scores['Mean_Precision']:.3f}")

save_prediction_table(
    build_prediction_rows("ml_decomposed", dec_spans),
    DECOMPOSED_OUTPUT_CSV,
)


# Summary
print("\n" + "=" * 60)
print("Summary Comparison")
print("=" * 60)
print(f"  {'Category':<8} {'Model':<15} {'Mean Precision':>15}")
print(f"  {'─' * 40}")

for category in ["Pop", "Int", "Out"]:
    for model_name, spans in [("End-to-End", e2e_spans),
                              ("Decomposed", dec_spans)]:
        scores = score_predictions(test_texts, test_masks,
                                   spans[category], category)
        print(f"  {category:<8} {model_name:<15} {scores['Mean_Precision']:>15.3f}")

print("\nAll experiments complete.")
