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
