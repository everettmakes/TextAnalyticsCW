import numpy as np
from pathlib import Path

# NOTE: Update DATA_ROOT to your local data folder before running
DATA_ROOT = r"D:\Pycharm\TextAnalyticsCW\data"

# Do not modify below
DATA_DIR = Path(DATA_ROOT) / "ebm_nlp_2_00"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


#  Load document IDs
def get_doc_ids(split="train", label_type="participants"):
    if split == "test":
        split = "test/gold"

    ann_dir = (DATA_DIR / "annotations" / "aggregated"
               / "hierarchical_labels" / label_type / split)

    doc_ids = [p.stem.split(".")[0] for p in ann_dir.glob("*.AGGREGATED.ann")]
    return sorted(doc_ids)


# Load tokens
def load_document(doc_id):
    doc_path = DATA_DIR / "documents" / f"{doc_id}.tokens"
    with open(doc_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_documents(doc_ids):
    return [load_document(doc_id) for doc_id in doc_ids]


#  Load labels

def load_labels_for_doc(doc_id, label_type="participants", split="train"):
    if split == "test":
        split = "test/gold"

    ann_path = (DATA_DIR / "annotations" / "aggregated"
                / "hierarchical_labels" / label_type / split
                / f"{doc_id}.AGGREGATED.ann")

    if not ann_path.exists():
        print(f"  [WARN] {ann_path} does not exist!")
        return None

    with open(ann_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_labels(doc_ids, label_type="participants", split="train"):
    labels = []
    for doc_id in doc_ids:
        doc_labels = load_labels_for_doc(doc_id, label_type, split)
        if doc_labels is not None:
            labels.append(doc_labels)
    return labels


# Convert labels to binary mask
def hierarchical_to_binary(tags):
    return [0 if int(t) == 0 else 1 for t in tags]


def convert_all_labels_to_binary(labels):
    return [hierarchical_to_binary(doc_labels) for doc_labels in labels]


# Main

print("Loading document IDs...")

# Get IDs present in all three categories (intersection)
p_ids = set(get_doc_ids("train", "participants"))
i_ids = set(get_doc_ids("train", "interventions"))
o_ids = set(get_doc_ids("train", "outcomes"))
train_ids = sorted(p_ids & i_ids & o_ids)

p_test_ids = set(get_doc_ids("test", "participants"))
i_test_ids = set(get_doc_ids("test", "interventions"))
o_test_ids = set(get_doc_ids("test", "outcomes"))
test_ids = sorted(p_test_ids & i_test_ids & o_test_ids)

print(f"  Train documents: {len(train_ids)}")
print(f"  Test documents : {len(test_ids)}")

# Load tokens
print("\nLoading tokens...")
train_tokens = load_documents(train_ids)
test_tokens = load_documents(test_ids)

# Load labels for all three categories
print("Loading labels...")
train_labels_p = load_labels(train_ids, "participants", "train")
train_labels_i = load_labels(train_ids, "interventions", "train")
train_labels_o = load_labels(train_ids, "outcomes", "train")

test_labels_p = load_labels(test_ids, "participants", "test")
test_labels_i = load_labels(test_ids, "interventions", "test")
test_labels_o = load_labels(test_ids, "outcomes", "test")

# Convert to binary masks
print("Converting labels to binary masks...")
train_labels_p = convert_all_labels_to_binary(train_labels_p)
train_labels_i = convert_all_labels_to_binary(train_labels_i)
train_labels_o = convert_all_labels_to_binary(train_labels_o)

test_labels_p = convert_all_labels_to_binary(test_labels_p)
test_labels_i = convert_all_labels_to_binary(test_labels_i)
test_labels_o = convert_all_labels_to_binary(test_labels_o)

# Join tokens into full abstract strings
train_texts = [" ".join(tokens) for tokens in train_tokens]
test_texts = [" ".join(tokens) for tokens in test_tokens]

# Save to .npz
output_file = PROCESSED_DIR / "ebm_abstracts_full.npz"

np.savez_compressed(
    output_file,
    train_texts=np.array(train_texts, dtype=object),
    train_p=np.array(train_labels_p, dtype=object),
    train_i=np.array(train_labels_i, dtype=object),
    train_o=np.array(train_labels_o, dtype=object),
    test_texts=np.array(test_texts, dtype=object),
    test_p=np.array(test_labels_p, dtype=object),
    test_i=np.array(test_labels_i, dtype=object),
    test_o=np.array(test_labels_o, dtype=object),
)

print(f"\nSaved to: {output_file}")