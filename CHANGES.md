# V2.1 fixes

## Bugs fixed

| # | Problem | Fix |
|---|---|---|
| 1 | Without `model.safetensors`, `_transformer_bundle()` passed its `config.json` check, failed in `from_pretrained`, and the bare `except: return None` hid the reason. `lru_cache` then kept the failure until a restart. | Checks for the weights file, records the error, shows it in the sidebar, and has a **Try loading the model again** button. Renames a single stray `*_model.safetensors` file automatically. |
| 2 | `Rs.`, `No.` and `Sec.` counted as sentence ends, so "Room rent is limited to Rs. 5,000 per day" vanished from Clause explorer (both halves were under 45 chars) and QA showed "...limited to Rs." | Abbreviation-aware splitter in `src/text_utils.py`, used by the app, QA, summary and `create_clause_csv`. |
| 3 | A wrapped line starting "2.5 lakh" was read as clause number "2." | Clause numbers must be followed by whitespace. |
| 4 | Summary never showed limits: the app read `"Limits"`, the module used `"Limits and cost sharing"`. | Same key in both places. |
| 5 | Summary listed "...is not covered" sentences under Coverage, also showed headings, and listed one sentence under several topics. | Negation filter, heading filter, and each sentence goes to its best-matching topic. |
| 6 | `policy_summary.py` contained the whole module twice. | Duplicate removed. |
| 7 | TF-IDF `stop_words="english"` removed *not / no / without / except*, so "What is not covered?" became "covered". | Custom stop-word list keeps negations. |
| 8 | Topic detection used substrings in the wrong direction (`"exclusion" in "excluded"` is False), so two of the app's own example questions got no topic boost. Only the first topic was boosted, and `"limit"` matched "unlimited". | Shared `src/topics.py` with word-boundary triggers; all detected topics are used. |
| 9 | QA topic phrases were matched against hyphen-stripped text, so "pre-existing disease", "sub-limit" and "co-payment" never matched. | Matched against the original lower-cased sentence. |
| 10 | The "Passages to show" slider did nothing above 3 (`_evidence(limit=3)` was hard-coded). | `answer_question(..., max_evidence=top_k)`. |
| 11 | `classify_blocks(_key, blocks)`: Streamlit does not hash `_`-prefixed arguments, so the document hash was ignored, the whole tuple was hashed on every rerun, and inference ran one paragraph at a time. | Hash the doc key, skip hashing the blocks, and use batched, length-sorted `predict_clauses()`. |
| 12 | The clause list length was not reset for a new document or filter. | Reset on upload and filter change. |
| 13 | Partly scanned PDFs: pages without text were dropped silently. | Warning with the number of skipped pages. |
| 14 | `pytest -q` failed with `No module named 'src'`. | `pytest.ini` (`pythonpath = .`). Tests went from 4 to 15. |
| 15 | `create_clause_csv.py` split on blank lines (PyMuPDF rarely produces them), giving one block per page; it and `batch_extract.py` used paths relative to the current folder. | Same `paragraphs()` as the app; paths from the project root. |
| 16 | `evaluate_on_testset.py` loaded the model separately from the app. | Uses the app's own loader and predictor, and refuses to score a fallback. |
| 17 | Retriever crashed with "empty vocabulary" on PDFs with no real words. | Handled. |
| 18 | Policy text containing `*` could turn into bold/italic markdown in answers. | Escaped. |
| 19 | `.gitignore` hid `data/annotations/` and the model folder; the 438 MB weights cannot go to GitHub without LFS. | `.gitignore` fixed; `.gitattributes` for Git LFS. |
| 20 | Seed `clauses.csv` used labels `claim_procedure` and `renewal_cancellation`, which are not among the 9 classes. | Renamed to `claims`, `renewal`. |
| 21 | Launchers never reinstalled after requirements changed. | Setup flag renamed so they reinstall once. |

## Removed or moved (useless or misleading)

- `_qa_pipeline` (roberta-squad2) was never called, but the README advertised it. Removed, and the README corrected.
- `sentence-transformers` was installed for semantic retrieval that does not exist. Removed. `datasets` and `accelerate` were only for the old scripts and moved to `legacy/requirements-legacy.txt`.
- DistilBERT `train_transformer.py` and `evaluate_model.py` cannot run on the shipped data. Moved to `legacy/`.
- `Passage.section` and the unused `sentences` argument removed. `.pytest_cache/` removed. `pytest` moved to `requirements-dev.txt`.
- Notebook outputs (`results_summary.csv`, `pool_labeled.csv`) moved from `models/` to `reports/colab/`.

## Documentation corrected

- README: 50 (not 45) inventory rows. The results table now compares against the fair baselines trained on the same data (TF-IDF 77.0%, frozen BGE 79.0%), not the 46.7% number. It also states that BGE and Legal-BERT are tied within ±4 points, that the model was chosen on the test set, and that train and test share documents.
- Notebook: "hand-labelled" changed to "LLM-labelled", and a caveat cell added.
- Confidence tags renamed from "High/Medium/Low confidence" to "Model fairly sure / unsure / guessing". On the test set they were right 89% / 48% / 35% of the time.
