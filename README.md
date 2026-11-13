# Malicious Prompt Detection – Data Preparation

CT118-3-3-ODL group assignment. This repository contains the EDA and data preparation for classifying LLM prompts as **benign (0)** or **malicious (1)** using a BiLSTM and a 1D-CNN.

## Repository structure

```
├── eda.ipynb                    # Exploratory data analysis on raw MPDD
├── data-preparation.ipynb       # Cleaning, splitting, tokenisation, external test set
└── cleaned-dataset/
    ├── train.csv                # Cleaned text, training set
    ├── validation.csv           # Cleaned text, validation set
    ├── test.csv                 # Cleaned text, test set
    ├── mpdd_prepared.npz        # Model-ready arrays (MPDD)
    └── safeguard_prepared.npz   # Model-ready arrays (external test set)
```

The raw `MPDD/` folder is excluded via `.gitignore`. To re-run the notebooks, download MPDD from [Kaggle](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd) into `MPDD/`. Safe-Guard is downloaded automatically through the Hugging Face `datasets` library.

## 1. What has been done

### EDA (raw MPDD, 39,234 prompts)
- No missing or empty prompts, and no conflicting labels. Classes are exactly balanced (19,617 each).
- 119 duplicates after lowercasing and trimming spaces; 4,035 prompts have a near-duplicate (TF-IDF cosine similarity ≥ 0.90).
- Benign prompts are short (max 61 words); malicious prompts range up to 8,445 words. 95% of prompts have ≤ 186 words.
- 67,462 unique words.
- **Shortcut risk:** 97.8% of benign prompts are Quora questions and 89.7% of malicious prompts come from PredictionGuard. 97.2% of benign prompts end with "?" compared with 18.7% of malicious prompts.

### Data preparation
1. **Cleaning:** lowercase, URLs replaced with `url`, emojis and non-ASCII characters removed, punctuation removed, extra spaces and line breaks merged.
2. **Removal:** 2 empty prompts, 238 duplicates and 1,347 near-duplicates removed, leaving **37,647 prompts** (51.9% benign, 48.1% malicious).
3. **Split:** stratified 70:15:15 with `random_state=42`, giving **26,352 / 5,647 / 5,648** prompts.
4. **Tokenisation:** Keras `TextVectorization` fitted on the training set only. Vocabulary of 20,000 words; every prompt padded or truncated to 200 tokens (the first 200 are kept).
5. **External test set:** Safe-Guard test split (2,060 prompts) cleaned the same way. 12 empty or duplicate prompts and 110 prompts overlapping with MPDD were removed, leaving **1,938 prompts** (72% safe, 28% attack).

## 2. Using the CSV files

The CSV files contain the **cleaned text** and labels. Use them to read prompts, for example when checking misclassified examples.

| Column | Description |
|---|---|
| `prompt` | Cleaned prompt text |
| `label` | 0 = benign, 1 = malicious |

```python
import pandas as pd

# keep_default_na=False stops prompts like "null" or "na" being read as missing values
test_df = pd.read_csv("cleaned-dataset/test.csv", keep_default_na=False)
```

Row *i* in each CSV matches row *i* in the corresponding NPZ array, so `test_df.iloc[i]` is the text of `X_test[i]`.

Do **not** re-split or re-tokenise the CSVs. Use the NPZ files for training so that both models receive identical inputs.

## 3. Using the NPZ files

The NPZ files contain **model-ready arrays**: each prompt is a sequence of 200 word IDs (0 = padding, 1 = unknown word).

```python
import numpy as np

# MPDD: training, validation and test sets
data = np.load("cleaned-dataset/mpdd_prepared.npz")
X_train, y_train = data["X_train"], data["y_train"]  # (26352, 200)
X_val, y_val = data["X_val"], data["y_val"]          # (5647, 200)
X_test, y_test = data["X_test"], data["y_test"]      # (5648, 200)

# Safe-Guard: external test set
ext = np.load("cleaned-dataset/safeguard_prepared.npz")
X_external, y_external = ext["X_external"], ext["y_external"]  # (1938, 200)
```

- Set the embedding layer to match the vocabulary: `Embedding(input_dim=20000, output_dim=...)`.
- Use the **validation set** for hyperparameter tuning and the **test set** only for final evaluation.
- Use `safeguard_prepared.npz` **once**, on the final tuned model. Because it is imbalanced (72:28), report precision, recall and F1 for the malicious class, not accuracy alone.

## Requirements

`pandas`, `scikit-learn`, `tensorflow`, `datasets`, `matplotlib`, `seaborn`