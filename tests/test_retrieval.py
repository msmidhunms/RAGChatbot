import pytest
from langchain_core.documents import Document

from rag_chatbot.core.types import RetrievedChunk
from rag_chatbot.generation.schemas import ExcerptScore, QueryList, RelevanceScores
from rag_chatbot.ingestion.manifest import Manifest
from rag_chatbot.ingestion.pipeline import IngestionPipeline
from rag_chatbot.retrieval.factory import RetrievalPipeline
from rag_chatbot.retrieval.hybrid import fuse
from rag_chatbot.retrieval.sparse import BM25Retriever, tokenize
from rag_chatbot.stores import build_store
from rag_chatbot.stores.docstore import SQLiteDocStore


def chunk(cid, score, text="t"):
    return RetrievedChunk(Document(page_content=text, metadata={"chunk_id": cid}), score)


# --------------------------------------------------------------------- fusion
def test_rrf_prefers_items_in_both_lists():
    a = [chunk("x", 0.9), chunk("y", 0.8), chunk("z", 0.7)]
    b = [chunk("z", 12.0), chunk("w", 3.0)]
    fused = fuse([a, b], method="rrf", rrf_k=60)
    assert fused[0].chunk_id == "z"
    assert [c.rank for c in fused] == [1, 2, 3, 4]
    assert {c.source for c in fused} == {"rrf"}


def test_rrf_weights_and_k():
    a = [chunk("x", 1), chunk("y", 1)]
    b = [chunk("y", 1), chunk("x", 1)]
    assert fuse([a, b], weights=[0.9, 0.1], k=1)[0].chunk_id == "x"
    assert fuse([a, b], weights=[0.1, 0.9], k=1)[0].chunk_id == "y"


def test_weighted_fusion_normalizes_scales():
    dense = [chunk("x", 0.95), chunk("y", 0.10)]
    sparse = [chunk("y", 40.0), chunk("x", 2.0)]  # very different scale
    fused = fuse([dense, sparse], weights=[0.5, 0.5], method="weighted")
    assert fused[0].score == pytest.approx(fused[1].score)
    assert fuse([dense, sparse], weights=[0.7, 0.3], method="weighted")[0].chunk_id == "x"
    assert fuse([[chunk("q", 3.0)], []], method="weighted")[0].score == pytest.approx(1.0)


# ----------------------------------------------------------------------- bm25
def test_bm25_and_rebuild(tmp_path):
    ds = SQLiteDocStore(tmp_path / "ds.sqlite")
    ds.put(
        [
            Document(page_content="espresso pressure nine bars", metadata={"chunk_id": "1", "src": "a"}),
            Document(page_content="pour over paper filter", metadata={"chunk_id": "2", "src": "b"}),
        ]
    )
    bm25 = BM25Retriever(ds)
    assert bm25.retrieve("espresso", 5)[0].chunk_id == "1"
    assert bm25.retrieve("espresso", 5, {"src": "b"}) == []
    assert bm25.retrieve("", 5) == []
    ds.put([Document(page_content="cold brew steeps for hours", metadata={"chunk_id": "3"})])
    assert bm25.retrieve("cold brew", 5)[0].chunk_id == "3"  # rebuilt after write
    assert tokenize("Café, NINE-bars!") == ["café", "nine", "bars"]


# ------------------------------------------------------------ full pipeline
@pytest.fixture
def setup(make_cfg, keyword_embeddings, corpus, tmp_path, fake_llm):
    def build(*overrides):
        cfg = make_cfg(
            "ingestion.cleaning.min_chars=10",
            "splitter.chunk_size=200",
            "splitter.chunk_overlap=20",
            *overrides,
        )
        store = build_store(cfg.vector_store, keyword_embeddings, namespace="kw", data_dir=cfg.app.data_dir)
        ds = SQLiteDocStore(tmp_path / f"ds-{len(overrides)}.sqlite")
        IngestionPipeline(cfg, store, ds, Manifest(None)).run([str(corpus)])
        return RetrievalPipeline(cfg, store, ds, keyword_embeddings, lambda: fake_llm)

    return build


@pytest.mark.parametrize("strategy", ["dense", "sparse", "hybrid"])
@pytest.mark.parametrize("search_type", ["similarity", "mmr"])
def test_strategies_find_answer(setup, strategy, search_type):
    rp = setup(f"retrieval.strategy={strategy}", f"retrieval.search_type={search_type}", "retrieval.k=3")
    results = rp.run("espresso pressure bars")
    assert results and len(results) <= 3
    assert "Espresso" in results[0].text
    assert [c.rank for c in results] == list(range(1, len(results) + 1))


def test_threshold_search(setup):
    rp = setup(
        "retrieval.strategy=dense", "retrieval.search_type=threshold", "retrieval.score_threshold=0.99"
    )
    assert rp.run("espresso pressure bars") == []


def test_parent_strategy_returns_parents(setup):
    rp = setup(
        "retrieval.strategy=parent",
        "retrieval.parent.parent_chunk_size=400",
        "retrieval.parent.child_chunk_size=120",
        "retrieval.k=2",
    )
    results = rp.run("Great Red Spot storm Jupiter")
    assert results[0].source == "parent"
    assert "Jupiter" in results[0].text and len(results[0].text) > 120


def test_default_and_request_filters_merge(setup):
    rp = setup(
        "retrieval.strategy=dense", "retrieval.search_type=similarity", "retrieval.filters={file_type: csv}"
    )
    results = rp.run("waterproof jacket")
    assert results and all(c.metadata["file_type"] == "csv" for c in results)
    overridden = rp.run("planets", flt={"file_type": "md"})  # request filter wins over the default
    assert overridden and all(c.metadata["file_type"] == "md" for c in overridden)


def test_multi_query_transform(setup, fake_llm):
    fake_llm.structured["QueryList"] = QueryList(queries=["hottest planet", "Venus temperature", "x", "y"])
    rp = setup("query_transform.type=multi_query", "query_transform.num_queries=2")
    queries = rp.transform("which planet is hottest?")
    assert queries == ["which planet is hottest?", "hottest planet", "Venus temperature"]
    results = rp.search(queries)
    assert {c.source for c in results} == {"rrf"} and "Venus" in results[0].text


@pytest.mark.parametrize("mode", ["rewrite", "hyde", "step_back"])
def test_text_transforms(setup, fake_llm, mode):
    fake_llm.responses = ["Venus surface temperature hottest planet"]
    rp = setup(f"query_transform.type={mode}")
    queries = rp.transform("hot planet?", "User: hi")
    if mode == "step_back":
        assert queries == ["hot planet?", "Venus surface temperature hottest planet"]
    else:
        assert queries == ["Venus surface temperature hottest planet"]
    assert "Conversation so far" in fake_llm.calls[-1][-1].content


def test_transform_override_mode(setup, fake_llm):
    fake_llm.responses = ["rewritten query"]
    rp = setup()
    assert rp.transform("q") == ["q"]
    assert rp.transform("q", mode="rewrite") == ["rewritten query"]


def test_llm_reranker(setup, fake_llm):
    rp = setup("reranker.type=llm", "reranker.top_n=2", "retrieval.k=4")
    chunks = rp.search(["coffee"])
    assert len(chunks) >= 2
    last = len(chunks)
    fake_llm.structured["RelevanceScores"] = RelevanceScores(
        scores=[ExcerptScore(id=last, score=9.5), ExcerptScore(id=1, score=1), ExcerptScore(id=99, score=10)]
    )
    out = rp.postprocess("coffee", chunks)
    assert len(out) == 2 and out[0].chunk_id == chunks[-1].chunk_id
    assert out[0].source == "rerank:llm" and out[0].score == 9.5


def test_cross_encoder_reranker(setup, monkeypatch):
    import rag_chatbot.retrieval.rerankers as rr

    class FakeCE:
        def predict(self, pairs):
            return [len(text) for _, text in pairs]

    monkeypatch.setattr(rr, "_cross_encoder", lambda model: FakeCE())
    rp = setup("reranker.type=cross_encoder", "reranker.top_n=1")
    chunks = rp.search(["coffee"])
    out = rp.postprocess("coffee", chunks)
    assert len(out) == 1 and len(out[0].text) == max(len(c.text) for c in chunks)


def test_cohere_requires_key(setup, monkeypatch):
    from rag_chatbot.core.exceptions import MissingCredentialsError

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="COHERE_API_KEY"):
        setup("reranker.type=cohere")


def test_redundant_and_embeddings_filters(setup):
    rp = setup("compression.type=redundant_filter", "compression.similarity_threshold=0.95")
    dup = [
        chunk("a", 1, "coffee espresso pressure"),
        chunk("b", 0.9, "coffee espresso pressure"),
        chunk("c", 0.8, "jupiter storm"),
    ]
    assert [c.chunk_id for c in rp.compressor("q", dup)] == ["a", "c"]

    rp2 = setup("compression.type=embeddings_filter", "compression.similarity_threshold=0.3")
    kept = rp2.compressor("coffee espresso", dup)
    assert [c.chunk_id for c in kept] == ["a", "b"]


def test_llm_extract(setup, fake_llm):
    fake_llm.responses = ["Espresso uses nine bars.", "NO_OUTPUT"]
    rp = setup("compression.type=llm_extract")
    out = rp.compressor("pressure?", [chunk("a", 1, "long espresso text"), chunk("b", 1, "unrelated")])
    assert [c.text for c in out] == ["Espresso uses nine bars."]
    assert out[0].metadata["chunk_id"] == "a"


def test_has_postprocess(setup):
    assert not setup().has_postprocess
    assert setup("compression.type=redundant_filter").has_postprocess


def test_flashrank_reranker(setup, monkeypatch):
    pytest.importorskip("flashrank")
    import rag_chatbot.retrieval.rerankers as rr

    seen = {}

    class FakeRanker:
        def rerank(self, request):
            seen["query"] = request.query
            # favour the longest passage, return in arbitrary order
            return [{"id": p["id"], "score": len(p["text"]) / 1000} for p in request.passages]

    monkeypatch.setattr(rr, "_flashrank", lambda model: FakeRanker())
    rp = setup("reranker.type=flashrank", "reranker.top_n=2", "retrieval.k=4")
    chunks = rp.search(["coffee"])
    out = rp.postprocess("coffee espresso", chunks)
    assert seen["query"] == "coffee espresso"
    assert len(out) == 2 and out[0].source == "rerank:flashrank"
    assert len(out[0].text) == max(len(c.text) for c in chunks)
    assert out[0].score >= out[1].score


def test_parent_strategy_on_non_parent_index_falls_back(
    make_cfg, keyword_embeddings, corpus, tmp_path, fake_llm
):
    """Switching to strategy=parent without re-ingesting must still return results."""
    base = ["ingestion.cleaning.min_chars=10", "splitter.chunk_size=200", "splitter.chunk_overlap=20"]
    dense_cfg = make_cfg(*base, "retrieval.strategy=dense")
    store = build_store(
        dense_cfg.vector_store, keyword_embeddings, namespace="kw", data_dir=dense_cfg.app.data_dir
    )
    ds = SQLiteDocStore(tmp_path / "ds.sqlite")
    IngestionPipeline(dense_cfg, store, ds, Manifest(None)).run([str(corpus)])
    parent_cfg = make_cfg(*base, "retrieval.strategy=parent")
    rp = RetrievalPipeline(parent_cfg, store, ds, keyword_embeddings, lambda: fake_llm)
    results = rp.run("espresso pressure bars")
    assert results and "Espresso" in results[0].text
