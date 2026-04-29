import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


TARGETS = {
    "P": "Participants",
    "I": "Interventions",
    "O": "Outcomes",
}

# Labels that indicate PICO content
PICO_LABELS = {"1", "2", "3", "4", "5", "6", "7"}


#  Data loading
def load_data(short, split="train"):
    suffix = "" if split == "train" else "_test"
    path   = rf"D:\Pycharm\TextAnalyticsCW\data\labeled_data_{short}{suffix}.npz"
    data   = np.load(path, allow_pickle=True)
    return list(data["texts"]), data["labels"]


# Label utilities

def binarise(labels):
    return np.where(np.isin(labels, list(PICO_LABELS)), "PICO", "NONE")


#  Model 1: End-to-End
def run_end_to_end(txt_train, txt_test, y_train, y_test):
# TF-IDF feature extraction
    tfidf   = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(txt_train)
    X_test  = tfidf.transform(txt_test)

    # Train and predict
    clf    = LinearSVC(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


# Model 2: Decomposed

def run_decomposed(txt_train, txt_test, y_train, y_test):
    # TF-IDF feature extraction shared between both steps
    tfidf   = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = tfidf.fit_transform(txt_train)
    X_test  = tfidf.transform(txt_test)

    # Step 1: Binary classification (PICO vs NONE)
    y_bin_train = binarise(y_train)

    clf1       = LinearSVC(class_weight="balanced", max_iter=2000)
    clf1.fit(X_train, y_bin_train)
    step1_pred = clf1.predict(X_test)

    # Step 2: Fine-grained classification (1/2/3/4)
    # Train only on true PICO sentences
    pico_train_mask = np.isin(y_train, list(PICO_LABELS))
    X_train_pico    = X_train[pico_train_mask]
    y_train_pico    = y_train[pico_train_mask]

    # Predict only on sentences Step 1 classified as PICO
    pico_test_mask = step1_pred == "PICO"
    X_test_pico    = X_test[pico_test_mask]

    # Build final predictions
    final_pred = np.full(len(y_test), "NONE", dtype=object)

    if X_train_pico.shape[0] > 0 and X_test_pico.shape[0] > 0:
        clf2 = LinearSVC(class_weight="balanced", max_iter=2000)
        clf2.fit(X_train_pico, y_train_pico)
        final_pred[pico_test_mask] = clf2.predict(X_test_pico)

    return final_pred, step1_pred


#  Evaluation

def evaluate(y_test, y_pred, model_name):

    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")


    all_labels = sorted(set(y_test) | set(y_pred))
    print(classification_report(y_test, y_pred,
                                labels=all_labels,
                                zero_division=0))

    # Confusion matrix
    print("Confusion Matrix (rows=true, cols=pred):")
    cm     = confusion_matrix(y_test, y_pred, labels=all_labels)
    header = f"{'':>8}" + "".join(f"{l:>8}" for l in all_labels)
    print(header)
    for lbl, row in zip(all_labels, cm):
        print(f"{lbl:>8}" + "".join(f"{v:>8}" for v in row))

    # Coverage vs Precision
    print("\nCoverage vs Precision Analysis:")
    n_extracted  = int((np.array(y_pred) != "NONE").sum())
    n_total      = len(y_pred)
    coverage_pct = 100 * n_extracted / n_total
    macro_prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec    = recall_score(   y_test, y_pred, average="macro", zero_division=0)
    macro_f1     = f1_score(       y_test, y_pred, average="macro", zero_division=0)

    print(f"  Sentences extracted : {n_extracted}/{n_total} ({coverage_pct:.1f}%)")
    print(f"  Macro Precision     : {macro_prec:.3f}")
    print(f"  Macro Recall        : {macro_rec:.3f}")
    print(f"  Macro F1            : {macro_f1:.3f}")


def evaluate_step1(y_test, step1_pred):
    y_bin_test = binarise(y_test)
    print("\n  [Step 1 — Binary: PICO vs NONE]")
    print(classification_report(y_bin_test, step1_pred,
                                labels=["PICO", "NONE"],
                                zero_division=0))


# Main

for short, name in TARGETS.items():
    print(f"\n{'═'*60}")
    print(f"  PICO Category: {name} ({short})")
    print(f"{'═'*60}")

    # Load data
    try:
        txt_train, y_train = load_data(short, "train")
        txt_test,  y_test  = load_data(short, "test")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Run data_preprocessing.py first.")
        continue

    print(f"  Train: {len(txt_train)} sentences | Test: {len(txt_test)} sentences")

    #  Model 1: End-to-End
    y_pred_e2e = run_end_to_end(txt_train, txt_test, y_train, y_test)
    evaluate(y_test, y_pred_e2e, f"Model 1 — End-to-End | {name}")

    # Model 2: Decomposed
    y_pred_dec, step1_pred = run_decomposed(txt_train, txt_test, y_train, y_test)
    evaluate_step1(y_test, step1_pred)
    evaluate(y_test, y_pred_dec, f"Model 2 — Decomposed | {name}")

    # Comparison
    print(f"\n  {'─'*40}")
    print(f"  Summary: {name}")
    print(f"  {'─'*40}")
    print(f"  {'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*40}")

    for model_name, y_pred in [("End-to-End", y_pred_e2e),
                                ("Decomposed", y_pred_dec)]:
        p = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"  {model_name:<20} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

print("\n\nAll experiments complete.")