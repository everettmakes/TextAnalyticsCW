# Task 2-2  PICO Information Extraction (ML Pipeline)
 
## Overview
This pipeline extracts Population, Intervention, and Outcome (PICO) elements
from clinical trial abstracts using machine learning.
 
**Design Axis: End-to-End vs Decomposed**
 
| Model | Type | Description |
|---|---|---|
| Model 1 | End-to-End | TF-IDF → LinearSVC → predict PICO label per sentence directly |
| Model 2 | Decomposed | Step 1 (PICO vs Non-PICO) → Step 2 (fine-grained per category) |

## Files
- `prepare_data.py` : Convert raw EBM-NLP data into `ebm_abstracts_full.npz`
- `pipeline.py` : End-to-End vs Decomposed model training and evaluation
- `generate_table.py` : Generate structured PICO extraction table as CSV

## Data Preparation
Data preprocessing is adapted from Josh's `prepare_data.ipynb` in the
[TextAnalyticsCW](https://github.com/everettmakes/TextAnalyticsCW) repository.
 
Run `prepare_data.py` to generate `ebm_abstracts_full.npz` before running the pipeline.

 ## Setup
Update `DATA_ROOT` in each file to your local data folder before running:

```python
DATA_ROOT = r"your/local/path/to/data"
```

## Run
```bash
python prepare_data.py     # Step 1: Generate ebm_abstracts_full.npz
python pipeline.py         # Step 2: Train and evaluate models (also generates cache)
python generate_table.py   # Step 3: Generate pico_table_ml.csv
```
> **Note**: Run `pipeline.py` before `generate_table.py`.
> It generates a sentence dataset cache that speeds up `generate_table.py` significantly.
