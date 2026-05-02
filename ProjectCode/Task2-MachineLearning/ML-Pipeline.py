import os
import csv
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATA_FILE = REPO_ROOT / "ebm_nlp_2_00" / "processed" / "ebm_abstracts_full.npz"
CACHE_FILE = SCRIPT_DIR / "token_dataset_cache.npz"
E2E_OUTPUT_CSV = SCRIPT_DIR / "ml_end_to_end_predictions.csv"
DECOMPOSED_OUTPUT_CSV = SCRIPT_DIR / "ml_decomposed_predictions.csv"

CONTEXT_WINDOW = 2

#  Load data
print("Loading dataset...")
data = np.load(DATA_FILE, allow_pickle=True)
train_texts = list(data["train_texts"])
test_texts = list(data["test_texts"])
train_masks = {"Pop": data["train_p"], "Int": data["train_i"], "Out": data["train_o"]}
test_masks = {"Pop": data["test_p"],  "Int": data["test_i"],  "Out": data["test_o"]}
test_ids = list(data["test_ids"]) if "test_ids" in data.files else [
    f"test_{i}" for i in range(len(test_texts))
]
print(f"Train abstracts: {len(train_texts)} | Test abstracts: {len(test_texts)}\n")


#  Token feature extraction
def token_features(tokens, i, window=CONTEXT_WINDOW):
    features = {}
    token = tokens[i]

    # Current token features
    features["token"] = token.lower()
    features["is_upper"] = int(token.isupper())
    features["is_title"] = int(token.istitle())
    features["is_digit"] = int(token.isdigit())
    features["token_length"] = len(token)

    # Context window features
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        j = i + offset
        if 0 <= j < len(tokens):
            features[f"context_{offset}"] = tokens[j].lower()
        else:
            features[f"context_{offset}"] = "<PAD>"

    return features


def build_token_dataset(texts, masks, split_name="Data"):
    all_features = []
    all_labels = {"Pop": [], "Int": [], "Out": []}
    doc_lengths = []   # number of tokens per document

    for idx, text in tqdm(enumerate(texts), total=len(texts),
                          desc=f"Building {split_name} token features"):
        tokens = text.split()
        n_tokens = len(tokens)
        doc_lengths.append(n_tokens)

        for i in range(n_tokens):
            all_features.append(token_features(tokens, i))

        for key in ["Pop", "Int", "Out"]:
            mask = list(masks[key][idx])
            # Align mask length with token count
            if len(mask) >= n_tokens:
                all_labels[key].extend(mask[:n_tokens])
            else:
                # Pad with 0 if mask is shorter
                all_labels[key].extend(mask + [0] * (n_tokens - len(mask)))

    return (all_features,
            {k: np.array(v, dtype=int) for k, v in all_labels.items()},
            doc_lengths)


# Load or build token dataset
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
    test_labels = {
        "Pop": cache["test_pop"],
        "Int": cache["test_int"],
        "Out": cache["test_out"],
    }
    print(f"Train tokens: {len(train_features)} | Test tokens: {len(test_features)}\n")

else:
    print("Building token-level datasets...")
    train_features, train_labels, train_doc_lengths = \
        build_token_dataset(train_texts, train_masks, "Train")
    test_features, test_labels, test_doc_lengths = \
        build_token_dataset(test_texts, test_masks, "Test")

    print(f"Train tokens: {len(train_features)} | Test tokens: {len(test_features)}\n")

    # Vectorize features
    print("Vectorizing features...")
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_features)
    X_test = vec.transform(test_features)

    # Save cache
    np.savez_compressed(CACHE_FILE,
        train_features=np.array(train_features, dtype=object),
        test_features=np.array(test_features,   dtype=object),
        train_doc_lengths=np.array(train_doc_lengths),
        test_doc_lengths=np.array(test_doc_lengths),
        train_pop=train_labels["Pop"],
        train_int=train_labels["Int"],
        train_out=train_labels["Out"],
        test_pop=test_labels["Pop"],
        test_int=test_labels["Int"],
        test_out=test_labels["Out"],
    )
    print(f"Cache saved to: {CACHE_FILE}\n")

# Vectorize if loaded from cache
if os.path.exists(CACHE_FILE):
    print("Vectorizing features...")
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_features)
    X_test = vec.transform(test_features)
    print("Done.\n")


# Extract phrases from token predictions
def extract_phrases_from_token_preds(pred_labels, texts, doc_lengths):
    spans = []
    tok_idx = 0

    for i, text in enumerate(texts):
        tokens = text.split()
        n_tokens = doc_lengths[i]
        doc_preds = pred_labels[tok_idx: tok_idx + n_tokens]
        tok_idx += n_tokens

        # Merge contiguous PICO tokens into phrases
        phrases = []
        current = []
        for token, pred in zip(tokens[:n_tokens], doc_preds):
            if pred == 1:
                current.append(token)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))

        spans.append(" ; ".join(phrases) if phrases else "null")

    return spans


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


def score_predictions(test_texts, test_masks, predicted_spans, category):
    statuses = []
    precisions = []

    for i, (text, span) in enumerate(zip(test_texts, predicted_spans)):
        status, prec = evaluate_extraction(text, span, test_masks[category][i])
        statuses.append(status)
        precisions.append(prec if prec is not None else 0.0)

    return {
        "Hit" : statuses.count("Hit (Relaxed)"),
        "Partial" : statuses.count("Partial Hit"),
        "Miss" : statuses.count("Miss"),
        "Hallucinated" : statuses.count("Hallucinated"),
        "Mean_Precision": np.mean(precisions),
    }


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



# Model 1: End-to-End
print("=" * 60)
print("Model 1 — End-to-End (Token-Level)")
print("=" * 60)
print("Hypothesis: train independent binary classifiers per PICO category")
print("at token level. Each token classified using context window features.")
print("Contiguous PICO tokens merged into phrases.\n")

e2e_spans = {}

for category in ["Pop", "Int", "Out"]:
    y_train = train_labels[category]
    y_test = test_labels[category]

    clf = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n  [{category}] Token-level Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-PICO", "PICO"],
                                zero_division=0))

    spans = extract_phrases_from_token_preds(y_pred, test_texts, test_doc_lengths)
    scores = score_predictions(test_texts, test_masks, spans, category)
    e2e_spans[category] = spans

    print(f" [{category}] Phrase Extraction Scores (vs word-level gt mask):")
    print(f" Hit: {scores['Hit']}  Partial: {scores['Partial']}  "
          f"Miss: {scores['Miss']}  Hallucinated: {scores['Hallucinated']}")
    print(f" Mean Precision: {scores['Mean_Precision']:.3f}")


save_prediction_table(
    build_prediction_rows("ml_token_end_to_end", e2e_spans),
    E2E_OUTPUT_CSV,
)


# Model 2: Decomposed
print("\n" + "=" * 60)
print("Model 2 — Decomposed (Token-Level)")
print("=" * 60)
print("Hypothesis: Step 1 filters non-PICO tokens, Step 2 classifies")
print("per category. Reduces noise but error propagates from Step 1.\n")

# Step 1: binary
y_train_bin = np.where(
    (train_labels["Pop"] == 1) |
    (train_labels["Int"] == 1) |
    (train_labels["Out"] == 1), 1, 0)

y_test_bin = np.where(
    (test_labels["Pop"] == 1) |
    (test_labels["Int"] == 1) |
    (test_labels["Out"] == 1), 1, 0)

print("  Training Step 1 (binary: any PICO vs non-PICO)...")
clf_step1 = LinearSVC(class_weight="balanced", max_iter=2000)
clf_step1.fit(X_train, y_train_bin)
step1_pred = clf_step1.predict(X_test)

print("  [Step 1 — Binary: PICO vs Non-PICO]")
print(classification_report(y_test_bin, step1_pred,
                            target_names=["Non-PICO", "PICO"],
                            zero_division=0))

# Step 2: per-category on predicted PICO tokens only
pico_train_mask = y_train_bin == 1
pico_test_mask = step1_pred == 1

X_train_pico = X_train[pico_train_mask]
X_test_pico = X_test[pico_test_mask]

dec_spans = {}

for category in ["Pop", "Int", "Out"]:
    y_train_cat = train_labels[category][pico_train_mask]

    clf_step2 = LinearSVC(class_weight="balanced", max_iter=2000)
    clf_step2.fit(X_train_pico, y_train_cat)
    step2_pred = clf_step2.predict(X_test_pico)

    # Combine
    final_pred = np.zeros(len(test_labels[category]), dtype=int)
    final_pred[pico_test_mask] = step2_pred

    print(f"\n  [{category}] Step 2 Token-level Classification Report:")
    print(classification_report(test_labels[category], final_pred,
                                target_names=["Non-PICO", "PICO"],
                                zero_division=0))

    spans  = extract_phrases_from_token_preds(final_pred, test_texts, test_doc_lengths)
    scores = score_predictions(test_texts, test_masks, spans, category)
    dec_spans[category] = spans

    print(f"  [{category}] Phrase Extraction Scores (vs word-level gt mask):")
    print(f"    Hit: {scores['Hit']}  Partial: {scores['Partial']}  "
          f"Miss: {scores['Miss']}  Hallucinated: {scores['Hallucinated']}")
    print(f"    Mean Precision: {scores['Mean_Precision']:.3f}")


save_prediction_table(
    build_prediction_rows("ml_token_decomposed", dec_spans),
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
