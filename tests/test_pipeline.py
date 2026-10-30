from src.clause_classifier import LABELS, keyword_classify, predict_clause, predict_clauses
from src.policy_summary import summarize_policy
from src.qa import answer_question
from src.retriever import Retriever, chunk_pages
from src.text_utils import paragraphs
from src.topics import detect_topics


def test_keyword_classifier_returns_label():
    label, score = keyword_classify("The policy excludes cosmetic treatment.")
    assert label == "exclusion"
    assert score > 0


def test_predict_clause_returns_known_label():
    label, conf, method = predict_clause("The waiting period for pre-existing diseases is 36 months.")
    assert label in LABELS
    assert 0.0 <= conf <= 1.0
    assert method


def test_predict_clauses_batch_matches_single():
    texts = ["The waiting period for pre-existing diseases is 36 months.",
             "Cosmetic surgery is excluded unless it follows an accident.",
             "Notify the TPA within 48 hours of an emergency admission."]
    batch = predict_clauses(texts)
    assert len(batch) == 3
    for text, (label, conf, _) in zip(texts, batch):
        single_label, single_conf, _ = predict_clause(text)
        assert label == single_label
        assert abs(conf - single_conf) < 1e-3
    assert predict_clauses([]) == []


def test_chunk_pages_preserves_page_number():
    chunks = chunk_pages([{"page": 7, "text": "waiting period " * 80}], chunk_words=30, overlap=5)
    assert chunks
    assert all(c.page == 7 for c in chunks)


def test_retriever_returns_matching_passage():
    passages = chunk_pages([{"page": 2, "text": "The waiting period for pre-existing diseases is described in this section."}])
    results = Retriever(passages).search("pre-existing disease waiting period", top_k=1)
    assert results and results[0][0].page == 2


def test_not_covered_finds_exclusions_not_coverage():
    pages = [
        {"page": 1, "text": "Hospitalisation expenses are covered up to the sum insured. Day care treatment is covered."},
        {"page": 2, "text": "Cosmetic surgery is not covered. Expenses for dental treatment are excluded."},
    ]
    results = Retriever(chunk_pages(pages)).search("What is not covered?", top_k=2)
    assert results[0][0].page == 2


def test_retriever_survives_text_without_words():
    assert Retriever(chunk_pages([{"page": 1, "text": "1 2 3 . . ."}])).search("claim") == []


def test_topics_detected_for_example_questions():
    assert "limit" in detect_topics("Is there a co-payment or deductible?")
    assert "exclusion" in detect_topics("What is not covered?")
    assert "limit" not in detect_topics("Is there an unlimited benefit?")
    assert set(detect_topics("What waiting period applies before a claim?")) == {"waiting", "claim"}


def test_summary_has_limits_key():
    pages = [{"page": 3, "text": "A co-payment of 20% applies to every claim for insured persons above 60."}]
    summary = summarize_policy(pages)
    assert summary["Limits"] and summary["Limits"][0]["page"] == 3


def test_answer_respects_evidence_limit():
    topics = ["hospital cash", "ambulance cover", "maternity benefit", "organ donor expenses",
              "domiciliary treatment", "ayush treatment"]
    pages = [{"page": i, "text": f"Renewal of {t}: the grace period for renewal of the {t} add-on is thirty days."}
             for i, t in enumerate(topics, start=1)]
    retrieved = Retriever(chunk_pages(pages)).search("What is the grace period for renewal?", top_k=6)
    assert len(answer_question("What is the grace period for renewal?", retrieved, max_evidence=5)["evidence"]) >= 2


def test_paragraphs_join_lines():
    text = "1. The insured must notify the insurer\nwithin 48 hours of admission.\n2. Documents must be sent within 30 days of discharge."
    blocks = paragraphs(text)
    assert len(blocks) == 2 and blocks[0].endswith("admission.")


def test_rupee_abbreviation_does_not_split_clause():
    text = "Room rent is limited to Rs. 5,000 per day for all insured persons.\nThe next clause starts here and is long enough."
    blocks = paragraphs(text)
    assert any("Rs. 5,000 per day" in b for b in blocks)


def test_wrapped_amount_line_is_not_a_new_clause():
    text = "Ambulance charges are payable up to a maximum limit of\n2.5 lakh per policy year for each insured person."
    assert paragraphs(text) == ["Ambulance charges are payable up to a maximum limit of 2.5 lakh per policy year for each insured person."]


def test_qa_keeps_rupee_amount_in_answer():
    pages = [{"page": 4, "text": "Room rent is limited to Rs. 5,000 per day. Other charges are payable as per actuals."}]
    retrieved = Retriever(chunk_pages(pages)).search("What is the room rent limit?", top_k=1)
    assert "Rs. 5,000 per day" in answer_question("What is the room rent limit?", retrieved)["answer"]


def test_summary_coverage_skips_not_covered_sentences():
    pages = [{"page": 1, "text": "Dental treatment is not covered under this policy. "
                                 "In-patient hospitalisation expenses are covered up to the sum insured."}]
    coverage = summarize_policy(pages)["Coverage"]
    assert coverage and all("not covered" not in c["text"] for c in coverage)
