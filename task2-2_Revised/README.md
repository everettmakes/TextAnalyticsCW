## Files
- `data_preprocessing.py` : Process EBM-NLP dataset, generate train/test .npz files
- `pipeline.py` : End-to-End vs Decomposed pipeline comparison
- `generate_table.py` : Generate structured PICO extraction table as CSV
 ## Setup
Update `DATA_ROOT` in each file to your local data folder before running:

```python
DATA_ROOT = r"your/local/path/to/data"
```

## Run
```bash
python data_preprocessing.py
python pipeline.py
python generate_table.py
```
