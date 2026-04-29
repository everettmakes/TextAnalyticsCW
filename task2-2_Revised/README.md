## Files
- `data_preprocessing.py` : Process EBM-NLP dataset, generate train/test .npz files
- `pipeline.py` : End-to-End vs Decomposed pipeline comparison
 ## Setup
Update the following paths in both files before running:
- `DATA_DIR`
- `BASE_LABEL_DIR`  
- `OUTPUT_DIR`

## Run
```bash
python data_preprocessing.py
python pipeline.py
```
