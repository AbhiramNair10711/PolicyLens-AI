# PolicyLens AI

**NLP + Machine Learning + Deep Learning for explaining Indian health
insurance policy documents.**

PolicyLens AI extracts policy text, classifies important clauses,
retrieves evidence, and answers questions with page-level citations. It
is an educational prototype, not insurance, medical, or legal advice.

## Features

-   Upload a health insurance policy PDF in the Streamlit UI.
-   Extract text with page numbers using PyMuPDF.
-   Classify clauses into categories such as coverage, exclusions,
    waiting periods, claims, limits, premiums, renewal, and definitions.
-   Classify clauses with a fine-tuned **BGE** (`BAAI/bge-base-en-v1.5`) sequence classifier
    (`models/clause-classifier/`). Falls back to a TF-IDF model, then keyword rules, if it cannot load;
    the sidebar says which one is active and why.
-   Retrieve relevant passages using TF-IDF with topic expansion (negation-aware, so
    "What is not covered?" finds exclusions).
-   Answer by ranking sentences from the retrieved passages (extractive, nothing generated), with page
    numbers and a short canned explanation of the topic. Users should verify that each passage
    actually answers their question.
-   A source inventory CSV with 50 candidate policy/document records to
    collect from official insurer catalogs.
-   The old DistilBERT scripts are kept in `legacy/` for reference only.

## Important dataset note

The repository does **not** bundle 45 copyrighted policy PDFs.
`data/policy_source_inventory.csv` is a curated acquisition checklist
with official source pages and candidate product names. Open the
official source page, download the current policy wording and/or
Customer Information Sheet (CIS), and record the exact
product/version/date. Availability and document URLs may change. Check
each insurer's terms and applicable copyright/licence before
redistributing documents. Do not commit personal policy schedules, claim
records, or other sensitive information.

The initial annotation file contains only small **illustrative
examples**, not ground truth from real policies. Replace/extend it with
clauses extracted from documents and manually verified by at least one
reviewer. Keep the policy ID, page number, exact source text, and
annotation provenance.

## Open the app with one click (Windows)

1.  Double-click **`PolicyLens.bat`**. The first time it sets everything up (about 5-10 minutes, needs
    internet and Python 3.10/3.11). After that it opens in a few seconds.
2.  Your browser opens PolicyLens by itself. Keep the black window open while you use the app; close it to stop.
3.  Optional: double-click **`Make_Desktop_Shortcut.bat`** once to put a **PolicyLens AI** icon on your Desktop.

On macOS, double-click `PolicyLens.command` instead (you may need to run `chmod +x PolicyLens.command` once).

## Quick start

### Requirements

-   Python 3.10 or 3.11 is recommended.
-   Git
-   Internet access on the first run to install packages. Nothing is downloaded from Hugging
    Face at run time: the BGE checkpoint is loaded from `models/clause-classifier/`.

### Windows PowerShell

``` powershell
git clone https://github.com/YOUR_USERNAME/PolicyLens-AI.git
cd PolicyLens-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-ml.txt

streamlit run app/streamlit_app.py
```

Replace `YOUR_USERNAME` with your GitHub username. If PowerShell blocks
virtual-environment activation, you can run the project's Python
directly or follow Microsoft's PowerShell execution-policy guidance.

`requirements-ml.txt` (torch, transformers, safetensors) is needed for the BGE classifier. Without
it the app still runs, using its TF-IDF or keyword fallback for clause tags.

## Add policy documents

1.  Use the source inventory: `data/policy_source_inventory.csv`.
2.  Download documents from the insurer's official catalog pages.
3.  Put local PDFs in `data/raw/` (the folder is git-ignored).
4.  Use the app's PDF uploader to try a document, or run the extraction
    utility:

``` bash
python -m src.extract_pdf data/raw/your_policy.pdf --output data/processed/your_policy.json
```

5.  Review extracted text and page numbers before using it for
    annotation.
6.  Add manually reviewed clauses to `data/annotations/clauses.csv`.

## Using the website

1.  Start the app with `streamlit run app/streamlit_app.py`.
2.  Open the local URL shown in the terminal (usually
    `http://localhost:8501`).
3.  Upload a selectable-text health insurance policy PDF in the upload
    area.
4.  Review the extracted text and page numbers. Ask questions in **Ask
    the policy** and check the cited evidence before relying on the
    answer.
5.  Use **Clause explorer** to inspect clause categories and **Extracted
    text** to verify the source wording.

### Example questions to ask

Try these after uploading a policy that contains the relevant sections:

**Waiting periods** - What are the waiting periods under this policy? -
What does the policy say about pre-existing diseases? - Are there
exceptions to the waiting period?

**Claims** - What documents are required to make a claim? - How soon
must claim documents be submitted? - How does reimbursement work under
this policy?

**Coverage and exclusions** - What treatments or conditions are
excluded? - What hospital expenses are covered? - Are there any limits
or sub-limits? - Does the policy mention a co-payment or deductible?

**Renewal and premium** - What is the renewal process? - What is the
grace period for renewal? - What does the policy say about premium
payments?

The answer depends on the specific uploaded document and how well its
text was extracted. If a question concerns a specific benefit, ask about
that benefit by name. Check the cited page and full clause, because a
policy may contain different rules for different sections or benefits.

## Clause annotation schema

`data/annotations/clauses.csv` uses these columns:

-   `text`: exact clause text
-   `label`: one of the supported clause labels
-   `policy_id`: internal document identifier, not a policyholder ID
-   `page`: PDF page number (1-indexed)
-   `reviewed`: `yes` only after human review

Supported labels: `claims`, `coverage`, `definition`, `exclusion`, `limit`, `other`, `premium`,
`renewal`, `waiting_period`.

Do not use the illustrative seed rows for a final evaluation. Add more
reviewed examples. Split train/test by **policy document**, not by
randomly splitting near-duplicate clauses.

## Train the ML classifier

``` bash
python -m src.train_classifier
```

This trains a TF-IDF + Logistic Regression model from rows marked `reviewed=yes`, splits by policy
document, scores it on the fixed test set and saves it to `models/clause_classifier.joblib`. It is
only used if BGE cannot load. The bundled `clauses.csv` has 9 illustrative, unreviewed rows, so the
script will stop until you add real reviewed clauses.

## Model checkpoint note

The BGE weights file is `models/clause-classifier/model.safetensors` (~438 MB). If it is missing,
copy it there (see `MODEL_SETUP_BGE.md`) and press **Try loading the model again** in the sidebar.
A single wrongly-named `*.safetensors` file in that folder (for example `1793302240612_model.safetensors`
from a browser download) is renamed automatically. For GitHub, the file must go through Git LFS
(`.gitattributes` is already set up).

## Selected classifier: BGE (Google Colab workflow)

The selected classifier is trained in Google Colab (free GPU) with
`PolicyLens_Classifier_Colab.ipynb`. The notebook compares BGE and Legal-BERT as well as traditional baselines. In this V2 setup, BGE is intended to be the deployed classifier and Legal-BERT is a comparison model:

1.  A fixed 300-row test set (`data/eval/test_set_labeled.csv`) is held out and never trained on.
2.  Gemini labels a few thousand extra clause paragraphs for training; its agreement with the
    test-set labels is measured first (about 86%).
3.  TF-IDF, frozen-embedding and fine-tuned transformer models (Legal-BERT, BGE) are compared on
    the same test set.
4.  The fine-tuned BGE checkpoint is downloaded as `clause-classifier.zip`. Unzip it so that the folder
    `models/clause-classifier/` exists, then run:

``` bash
python -m src.evaluate_on_testset
```

This writes `reports/evaluation.md` with accuracy, macro-F1, per-class scores and a confusion
matrix. The app loads `models/clause-classifier` automatically.

**Label note:** all training and test labels except the 20 originally human-reviewed rows were
produced by LLMs (Claude for the test set, Gemini for training). Reported scores therefore mean
agreement with LLM-made labels, not human-verified ground truth.

## Run tests

``` bash
pip install -r requirements-dev.txt
pytest -q
```

## Project structure

``` text
PolicyLens-AI/
├── app/streamlit_app.py
├── data/
│   ├── policy_source_inventory.csv
│   ├── raw/                  # local PDFs; ignored by Git
│   ├── processed/            # generated text; ignored by Git
│   └── annotations/clauses.csv
├── models/clause-classifier/ # BGE checkpoint (model.safetensors via Git LFS)
├── reports/                  # evaluation.md, colab/ notebook outputs
├── src/
│   ├── extract_pdf.py, batch_extract.py, create_clause_csv.py
│   ├── text_utils.py         # paragraph + sentence splitting shared by everything
│   ├── topics.py             # topic vocabulary shared by retrieval and QA
│   ├── clause_classifier.py
│   ├── retriever.py
│   ├── qa.py
│   ├── policy_summary.py
│   ├── train_classifier.py
│   └── evaluate_on_testset.py
├── legacy/                   # old DistilBERT scripts (not used)
├── tests/
├── requirements.txt, requirements-ml.txt, requirements-dev.txt
└── README.md
```

## Current prototype status and model evaluation

The prototype demonstrates PDF text extraction, evidence retrieval, page references,
simple-English explanations, and clause classification into nine labels.

Classification results on the fixed 300-row test set (from `reports/colab/results_summary.csv`; all
models below were trained on the same Gemini-labelled data):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Keyword rules | 46.3% | - |
| TF-IDF + Logistic Regression | 77.0% | 0.745 |
| Frozen BGE embeddings + Logistic Regression | 79.0% | 0.778 |
| Fine-tuned Legal-BERT | 82.7% | 0.797 |
| **Fine-tuned BGE (selected)** | **82.3%** | **0.805** |
| *Gemini labels vs. test labels (reference)* | *86%* | - |

How to read this honestly:

-   Fine-tuning adds about 5 points over a TF-IDF model on the same data. BGE and Legal-BERT differ
    by one test row in accuracy; with 300 rows the 95% interval is about ±4 points, so they are tied.
    BGE was kept, but it did not clearly "win".
-   The final model was chosen by test-set macro-F1 in the notebook, and train/test paragraphs come
    from overlapping policy documents, so 82% is probably optimistic for an unseen insurer.
-   The test set is not a random sample: it has 30 rows per keyword-guess class plus 60 `other`.
-   The model's confidence is informative but not a guarantee: at ≥0.80 it was right 89% of the time
    (258 rows), at 0.55–0.80 48%, and below 0.55 35%.
-   The training CSV used by the notebook (with `row_id` and `suggested_label`) is not in this repo,
    so training cannot be reproduced from here.
-   The old DistilBERT run (28.95% accuracy on 38 examples) is not comparable.

Known limits: labels are LLM-made and noisy (Gemini agreed with the test labels 86% of the
time); many paragraphs are broken fragments or fit more than one label; scanned PDFs need OCR.
The app no longer applies hand-written rule overrides, so what it shows is the model's output.

## Evaluation plan

-   Classification: macro-F1, per-class precision/recall, confusion
    matrix.
-   Extraction: field-level precision/recall/F1 and exact match for
    durations/amounts.
-   Retrieval: Recall@5 and MRR on manually authored questions with
    known source pages.
-   QA: answer correctness, evidence support, citation accuracy, and
    correct abstention when the document does not answer the question.
-   OCR: manually inspect a sample of pages, especially tables,
    exclusions, and footnotes.

## Safety, privacy, and limitations

-   This app explains text from the uploaded policy; it does not
    determine whether an insurer will approve a claim.
-   Coverage may depend on endorsements, schedule details, waiting
    periods, exclusions, sub-limits, and facts not included in the
    wording.
-   Never upload or commit sensitive personal documents without consent
    and appropriate security controls.
-   Avoid sharing user-uploaded PDFs with third-party APIs unless users
    have explicitly consented.
-   Always verify important decisions against the full policy and the
    insurer/qualified professional.
-   Generated answers can be wrong. Citations and evidence must be
    checked.

## Data sources

Start with the official catalogs listed in
`data/policy_source_inventory.csv`, including: - IRDAI Health Insurance
Products portal: https://irdai.gov.in/health-insurance-products - HDFC
ERGO health policy wordings:
https://www.hdfcergo.com/download/policy-wordings/health - HDFC ERGO
health Customer Information Sheets:
https://www.hdfcergo.com/download/cis/health - Star Health downloads:
https://www.starhealth.in/downloads/ - SBI General downloads:
https://www.sbigeneral.in/downloads

Document availability, versions, and reuse rights vary. Prefer the exact
wording corresponding to the product/version under study; CIS documents
are useful for comparison and validation but are not a substitute for
full policy wording.

## License

Code is provided for educational use. This code license does not grant
rights to third-party policy documents. Add a LICENSE file before public
release and choose a licence appropriate for your project.
