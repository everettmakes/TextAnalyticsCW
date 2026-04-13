import os
import spacy
import numpy as np

# Base path configuration
DATA_DIR = "./data/ebm_nlp_2_00/documents"
BASE_LABEL_DIR = "./data/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels"

TARGETS = ["participants", "interventions", "outcomes"]

print("Loading Spacy model... ")
nlp = spacy.load("en_core_web_md")


def build_train_dataset(label_dir):
    # Construct train directory path
    train_dir = os.path.join(label_dir, "train")

    # Check if the directory exists
    if not os.path.exists(train_dir):
        return [], [], []

    # Get all files in the directory
    all_files = os.listdir(train_dir)
    label_files = []
    for f in all_files:
        if f.endswith(".ann") or f.endswith(".tags"):
            label_files.append(f)

    all_vectors = []
    all_texts = []
    all_labels = []

    for file_name in label_files:
        # Extract ID by splitting the string
        doc_id = file_name.split('.')[0]
        token_file = os.path.join(DATA_DIR, doc_id + ".tokens")
        label_file_path = os.path.join(train_dir, file_name)

        if not os.path.exists(token_file):
            continue

        # Open files and read lines
        f_tok = open(token_file, "r", encoding="utf-8")
        f_lab = open(label_file_path, "r", encoding="utf-8")

        tokens = []
        for line in f_tok:
            tokens.append(line.strip())

        labels = []
        for line in f_lab:
            labels.append(line.strip())

        f_tok.close()
        f_lab.close()

        # length check
        if len(tokens) != len(labels):
            continue

        current_sentence_tokens = []
        current_sentence_labels = []

        for i in range(len(tokens)):
            token = tokens[i]
            label = labels[i]

            current_sentence_tokens.append(token)
            current_sentence_labels.append(label)

            # Split by punctuation
            if token == "." or token == "?" or token == "!":
                if len(current_sentence_tokens) > 4:
                    sent_text = " ".join(current_sentence_tokens)
                    sent_vector = nlp(sent_text).vector


                    counts = {}
                    for l in current_sentence_labels:
                        if l != '0' and l != 'NONE' and l != 'O':
                            if l in counts:
                                counts[l] = counts[l] + 1
                            else:
                                counts[l] = 1

                    # Find the label with the highest count
                    final_label = 'NONE'
                    max_count = 0
                    for l in counts:
                        if counts[l] > max_count:
                            max_count = counts[l]
                            final_label = l

                    all_vectors.append(sent_vector)
                    all_texts.append(sent_text)
                    all_labels.append(final_label)

                current_sentence_tokens = []
                current_sentence_labels = []

    return np.array(all_vectors), all_texts, np.array(all_labels)


# Main execution loop
for target in TARGETS:
    print("\n" + "=" * 40)
    print("Processing: " + target)
    print("=" * 40)

    current_label_path = os.path.join(BASE_LABEL_DIR, target)
    save_path = "./data/labeled_data_" + target[0].upper() + ".npz"

    # Execute dataset building
    vectors, texts, labels = build_train_dataset(current_label_path)

    if len(vectors) > 0:
        np.savez_compressed(save_path, vectors=vectors, texts=texts, labels=labels)
        print("Success. File saved to: " + save_path)
        print("Total sentences: " + str(len(vectors)))
