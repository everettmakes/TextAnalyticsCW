**Task 2: Structured Information Extraction**

By converting unstructured descriptions of clinical trials to structured tables, we can help biomedical
researchers to locate and synthesise valuable evidence across many different studies.

Objective: Given a corpus of clinical trial abstracts, extract information into a predefined table
schema. Evaluate how different extraction strategies affect the quality of the extracted data.

The text data and table schema are provided by the EBM-NLP dataset. The dataset consists of clinical
trial abstracts, and the task is to extract the population, intervention, comparator and outcomes of
the trial from the abstract. A script for loading the data will be provided in the Text Analytics Github
repository.

**Student group work:**
1. As a preliminary consider clustering approaches: Before extraction, use k-means or
Hierarchical clustering (HAC) on sentence embeddings to see whether natural clusters
correspond to schema fields.
2. Design and implement an information extraction pipeline:
a. Choose two important design axes to explore, such as: rule-based approaches versus
LLMs; end-to-end models versus decomposing the task into smaller steps; varying
training and zero/few-shot learning strategies.
b. For each chosen approach, design a complete pipeline for extracting the tabular
data.
c. For each design axis, compare 2-3 alternative approaches, motivated by a clear
hypothesis about how each approach will perform.
3. Evaluate each variant of your pipeline in terms of:
a. Field-level accuracy/F1/etc.
b. Coverage vs precision,
c. Downstream usability (e.g., can queries be answered).
4. Discuss trade-offs between approaches and the limitations of your approach


### **Task 2: Clinical Data Extraction**

**Goal:** Convert medical abstracts into a structured **PICO** table (Population, Intervention, Comparator, Outcome).

---

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

---
