### **Task 2: Clinical Data Extraction**

**TODO:** Convert medical abstracts into a structured **PICO** table (Population, Intervention, Comparator, Outcome).

-----

### **1. Preliminary (Clustering)**
* Test if **K-means** or **HAC** can naturally group sentences into the PICO categories using sentence embeddings.

### **2. Development (The Pipeline)**
Pick **two axes** to test (e.g., LLMs vs. Rules) and build pipelines for each. Compare 2–3 variants per axis based on a specific hypothesis.

### **3. Evaluation**
* **Metrics:** F1-score, Accuracy.
* **Balance:** Precision vs. Recall (Coverage).
* **Utility:** Can the resulting table actually answer research queries?

### **4. Discussion**
* Analyze trade-offs (speed, cost, accuracy) and identify limitations.

-----


-----

# 📑 Group Project: Clinical Information Extraction (EBM-NLP)

## 0. Project Overview
* **Main Challenge:** Converting unstructured clinical trial abstracts into structured PICO tables.
* **Status:** In Progress 🚧

------

## 1. Abstract (0.25 Pages)
- [ ] Summarize the PICO extraction challenge.
- [ ] List the primary ideas/models tested.
- [ ] State high-level experimental findings.

## 2. Introduction (1.5 Pages)
- [ ] **Task Outline:** Explain the transition from raw text to structured data.
- [ ] **Motivation:** Why do biomedical researchers need this?
- [ ] **Data Exploration:** - [ ] Include examples from the EBM-NLP dataset.
  - [ ] Insert data distribution plots (e.g., average sentence length, class balance).

## 3. Methods (3.5 Pages)
### 3a. Baseline System
* **Preprocessing:** Cleaning, tokenization, and sentence splitting.
* **Text Representation:** Word embeddings, TF-IDF, or Sentence-BERT.
* **The Model:** Explain the initial extraction logic (include equations for loss functions or similarity metrics if applicable).

### 3b. Design Axes & Improvements
* **Axis 1:** (e.g., Rule-based vs. LLM) - Describe the hypothesis.
* **Axis 2:** (e.g., End-to-end vs. Decomposed) - Describe the hypothesis.
* **Implementation:** Detail the architecture of each proposed variant.

## 4. Experimental Evaluation (3.5 Pages)
### 4a. Metrics & Setup
* **Metrics:** Define F1-score, Precision, and Recall. Note their limitations in medical contexts.
* **Setup:** Detail dataset splits (Train/Val/Test) and Hyperparameter tuning.
* **Libraries:** Link to HuggingFace models or core NLP libraries used.

### 4b. Results & Error Analysis
* **Comparison Table:** Compare all model variants.
* **Visuals:** Precision-Recall curves or Error Heatmaps.
* **Error Analysis:** What specific text patterns (e.g., long sentences, medical jargon) caused the most failures?

## 5. Conclusions (1 Page)
- [ ] Summary of what worked (and what didn't).
- [ ] Final insights on the trade-off between complexity and accuracy.
- [ ] List remaining open challenges in biomedical NLP.

-----
