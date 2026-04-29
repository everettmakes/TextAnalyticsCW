import os
import numpy as np

# NOTE: Update DATA_ROOT to your local data folder before running
DATA_ROOT = r"D:\Pycharm\TextAnalyticsCW\data"

# Do not modify below
DATA_DIR = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "documents")
BASE_LABEL_DIR = os.path.join(DATA_ROOT, "ebm_nlp_2_00", "annotations", "aggregated", "hierarchical_labels")
OUTPUT_DIR = DATA_ROOT
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PICO categories
TARGETS = ["participants", "interventions", "outcomes"]

# Use gold standard annotations for test split
SPLITS = ["train", "test/gold"]


#  Core function
def build_dataset(label_dir, split):
    split_dir = os.path.join(label_dir, split)

    # Check directory exists
    if not os.path.exists(split_dir):
        print(f"  [WARN] Directory not found : {split_dir}")
        return [], np.array([])

    # Collect annotation files
    label_files = [f for f in os.listdir(split_dir)
                   if f.endswith(".ann") or f.endswith(".tags")]

    all_texts = []
    all_labels = []

    for file_name in label_files:
        doc_id = file_name.split(".")[0]
        token_path = os.path.join(DATA_DIR, doc_id + ".tokens")
        label_path = os.path.join(split_dir, file_name)

        # Skip if token file is missing
        if not os.path.exists(token_path):
            continue

        with open(token_path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f]
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f]

        # Skip malformed files where lengths don't match
        if len(tokens) != len(labels):
            continue

        #Sentence segmentation on punctuation
        sent_tokens = []
        sent_labels = []

        for token, label in zip(tokens, labels):
            sent_tokens.append(token)
            sent_labels.append(label)

            # Split on sentence-ending punctuation
            if token in {".", "?", "!"}:
                # Ignore very short spans
                if len(sent_tokens) > 4:
                    sent_text = " ".join(sent_tokens)

                    # Majority-vote label: ignore O/NONE/0, take most frequent
                    counts = {}
                    for lbl in sent_labels:
                        if lbl not in {"0", "O", "NONE", ""}:
                            counts[lbl] = counts.get(lbl, 0) + 1

                    final_label = max(counts, key=counts.get) if counts else "NONE"

                    all_texts.append(sent_text)
                    all_labels.append(final_label)

                # Reset buffers for next sentence
                sent_tokens = []
                sent_labels = []

    return all_texts, np.array(all_labels)


#  Main loop
for target in TARGETS:
    short = target[0].upper()
    print("=" * 50)
    print(f"Processing: {target} ({short})")
    print("=" * 50)

    label_dir = os.path.join(BASE_LABEL_DIR, target)

    for split in SPLITS:
        texts, labels = build_dataset(label_dir, split)

        # Skip if no data found
        if len(texts) == 0:
            print(f"  [{split}] No data found, skipping. / 未找到数据，跳过。")
            continue

        # Determine output filename suffix
        suffix = "" if split == "train" else "_test"
        save_path = os.path.join(OUTPUT_DIR, f"labeled_data_{short}{suffix}.npz")

        np.savez_compressed(save_path,
                            texts=np.array(texts, dtype=object),
                            labels=labels)

        label_dist = {lbl: int((labels == lbl).sum()) for lbl in np.unique(labels)}
        print(f"  [{split}] Saved {len(texts):,} sentences → {save_path}")
        print(f"           Label distribution : {label_dist}")

print("\nPreprocessing complete.")
 