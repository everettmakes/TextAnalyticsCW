import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Define classification targets
targets = ["P", "I", "O"]
target_names = {"P": "Participants", "I": "Interventions", "O": "Outcomes"}

for t in targets:
    print("\nEvaluating Category: " + target_names[t])

    # Load preprocessed files
    train_path = "./data/labeled_data_" + t + ".npz"
    test_path = "./data/labeled_data_" + t + "_test.npz"

    data_train = np.load(train_path)
    data_test = np.load(test_path)


    # Axis 2: Strategy A - Simple Pipeline (TF-IDF)
    print("Running Strategy A: Simple Pipeline (TF-IDF)...")

    tfidf = TfidfVectorizer()
    X_train_simple = tfidf.fit_transform(data_train['texts'])
    X_test_simple = tfidf.transform(data_test['texts'])

    # Use balanced weights to handle class imbalance
    clf_simple = LinearSVC(class_weight='balanced')
    clf_simple.fit(X_train_simple, data_train['labels'])
    y_pred_simple = clf_simple.predict(X_test_simple)

    print("\n[Strategy A Report]")
    print(classification_report(data_test['labels'], y_pred_simple))

    print("\n[Strategy A Confusion Matrix]")
    print(confusion_matrix(data_test['labels'], y_pred_simple, labels=['1', '2', '3', '4', 'NONE']))

    # Axis 2: Strategy B - Complex Pipeline (SpaCy Vectors)
    print("Running Strategy B: Complex Pipeline (SpaCy Vectors)...")

    X_train_complex = data_train['vectors']
    X_test_complex = data_test['vectors']

    clf_complex = LinearSVC(class_weight='balanced')
    clf_complex.fit(X_train_complex, data_train['labels'])
    y_pred_complex = clf_complex.predict(X_test_complex)

    print("\n[Strategy B Report]")
    print(classification_report(data_test['labels'], y_pred_complex))
    print("\n[Strategy B Confusion Matrix]")
    print(confusion_matrix(data_test['labels'], y_pred_complex, labels=['1', '2', '3', '4', 'NONE']))