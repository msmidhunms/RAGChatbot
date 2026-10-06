import pytest

from rag_chatbot.core.exceptions import RAGError
from rag_chatbot.evaluation.dataset import load_dataset, save_dataset
from rag_chatbot.evaluation.generate import DatasetGenerator, allocate, is_near_duplicate
from rag_chatbot.generation.schemas import GeneratedFollowUp, GeneratedQA, GeneratedQuestion
from rag_chatbot.pipeline import RAGPipeline


@pytest.fixture
def pipe(make_cfg, keyword_embeddings, fake_llm, corpus):
    cfg = make_cfg("ingestion.cleaning.min_chars=10", "splitter.chunk_size=400", "splitter.chunk_overlap=0")
    p = RAGPipeline(cfg, llm=fake_llm, embeddings=keyword_embeddings)
    p.ingest([str(corpus)])
    return p


def _quote_from_prompt(messages, marker="Excerpt:\n"):
    """Return the first 6 words of the excerpt in the prompt, i.e. a valid verbatim quote."""
    text = messages[-1].content.split(marker, 1)[1]
    return " ".join(text.split()[:6])


def test_allocate():
    assert allocate(20, ["factual", "multi_hop", "unanswerable", "conversational"]) == {
        "factual": 10,
        "multi_hop": 4,
        "unanswerable": 3,
        "conversational": 3,
    }
    assert sum(allocate(7, ["factual", "unanswerable"]).values()) == 7
    with pytest.raises(RAGError, match="cannot generate"):
        allocate(5, ["filtered"])


def test_near_duplicates():
    assert is_near_duplicate("What is the flight time of the drone?", ["what is the drone flight time"])
    assert not is_near_duplicate("What is the flight time?", ["Who founded the company?"])


def test_generates_valid_factual_items(pipe, fake_llm):
    counter = iter(range(100))

    def factual(messages):
        n = next(counter)
        return GeneratedQA(
            question=f"Question topic{n} detail{n} item{n}?",
            answer="An answer.",
            evidence=[_quote_from_prompt(messages)],
        )

    fake_llm.structured["GeneratedQA"] = factual
    result = DatasetGenerator(pipe, seed=1).generate(4, ["factual"])
    assert len(result.items) == 4 and not result.rejected
    sources = {i.expected_sources[0] for i in result.items}
    assert len(sources) >= 3  # round-robin across documents
    assert all(i.generated and i.category == "factual" and i.evidence for i in result.items)


def test_rejects_invented_evidence_and_duplicates(pipe, fake_llm):
    attempts = iter(range(100))

    def factual(messages):
        n = next(attempts)
        if n == 0:
            return GeneratedQA(
                question="A hallucinated question here?", answer="x", evidence=["not in the text"]
            )
        if n == 1:
            return GeneratedQA(
                question="Duplicate question about apples?",
                answer="x",
                evidence=[_quote_from_prompt(messages)],
            )
        if n == 2:
            return GeneratedQA(
                question="Duplicate question about apples?",
                answer="x",
                evidence=[_quote_from_prompt(messages)],
            )
        return GeneratedQA(
            question=f"Distinct topic{n} detail{n}?", answer="x", evidence=[_quote_from_prompt(messages)]
        )

    fake_llm.structured["GeneratedQA"] = factual
    result = DatasetGenerator(pipe).generate(2, ["factual"])
    assert len(result.items) == 2
    assert any("verbatim" in r for r in result.rejected) and any(
        "near-duplicate" in r for r in result.rejected
    )


def test_all_categories_and_roundtrip(pipe, fake_llm, tmp_path):
    counter = iter(range(1000))

    def factual(messages):
        n = next(counter)
        return GeneratedQA(
            question=f"Single topic{n} detail{n}?", answer="a", evidence=[_quote_from_prompt(messages)]
        )

    def multi(messages):
        content = messages[-1].content
        a = " ".join(content.split("Excerpt A:\n", 1)[1].split()[:5])
        b = " ".join(content.split("Excerpt B:\n", 1)[1].split()[:5])
        n = next(counter)
        return GeneratedQA(question=f"Combined topic{n} detail{n}?", answer="ab", evidence=[a, b])

    fake_llm.structured["GeneratedQA"] = lambda m: multi(m) if "Excerpt A:" in m[-1].content else factual(m)
    fake_llm.structured["GeneratedQuestion"] = lambda m: GeneratedQuestion(
        question=f"Unknowable{next(counter)} thing here?"
    )
    fake_llm.structured["GeneratedFollowUp"] = lambda m: GeneratedFollowUp(
        first_question="Tell me about the subject.", follow_up=f"What about its part{next(counter)} please?"
    )
    result = DatasetGenerator(pipe, seed=3).generate(
        10, ["factual", "multi_hop", "unanswerable", "conversational"]
    )
    cats = [i.category for i in result.items]
    assert cats.count("factual") == 5 and cats.count("multi_hop") == 2
    assert cats.count("unanswerable") == 2 and cats.count("conversational") == 1
    multi_hop = next(i for i in result.items if i.category == "multi_hop")
    assert len(multi_hop.expected_sources) == 2 and len(multi_hop.evidence) == 2
    conv = next(i for i in result.items if i.category == "conversational")
    assert conv.history == ["Tell me about the subject."]
    path = save_dataset(result.items, tmp_path / "gen.jsonl")
    assert [i.id for i in load_dataset(path)] == [i.id for i in result.items]


def test_empty_index(make_cfg, keyword_embeddings, fake_llm):
    pipe = RAGPipeline(make_cfg(), llm=fake_llm, embeddings=keyword_embeddings)
    with pytest.raises(RAGError, match="index is empty"):
        DatasetGenerator(pipe)
