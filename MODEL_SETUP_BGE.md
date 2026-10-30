# Model setup (BGE)

The app's clause classifier is a fine-tuned `BAAI/bge-base-en-v1.5` **sequence-classification**
checkpoint (BERT architecture, 12 layers, hidden size 768, 9 labels) exported by
`PolicyLens_Classifier_Colab.ipynb`. A generic BGE embedding checkpoint will not work.

## Required files in `models/clause-classifier/`

| File | Included | Notes |
|---|---|---|
| `config.json` | yes | 9 labels, `BertForSequenceClassification` |
| `labels.json` | yes | label → id; must match `config.json` (checked at load time) |
| `tokenizer.json`, `tokenizer_config.json` | yes | |
| `model.safetensors` | add it | ~438 MB, 201 tensors |

If the weights file is missing or named differently, the sidebar shows the reason and the app uses the
TF-IDF model (if trained) or keyword rules. A single stray `*.safetensors` file in the folder is
renamed to `model.safetensors` automatically. After adding the file, press **Try loading the model
again** in the sidebar (no restart needed).

This checkpoint was verified to reproduce accuracy 0.8233 / macro-F1 0.8046 on
`data/eval/test_set_labeled.csv`.

## Run

```powershell
python -m pip install -r requirements.txt -r requirements-ml.txt
python -m src.evaluate_on_testset      # rewrites reports/evaluation.md
python -m streamlit run app/streamlit_app.py
```

`evaluate_on_testset` uses the same loading and prediction code as the app and refuses to run if BGE
did not load (it no longer silently scores a fallback).

## GitHub

`model.safetensors` is above GitHub's 100 MB limit. Run `git lfs install` once; `.gitattributes`
already routes `*.safetensors` through LFS, and `.gitignore` no longer hides `models/clause-classifier/`.

## Evaluation note

Test labels were made by an LLM (Claude), not humans, so scores mean agreement with those labels. See
the README for the other caveats (model chosen on the test set, overlapping documents).
