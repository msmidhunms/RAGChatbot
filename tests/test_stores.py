import importlib.util
import os

import pytest
from langchain_core.documents import Document

from rag.config.schema import VectorStoreConfig
from rag.core.exceptions import ConfigError
from rag.stores import build_store
from rag.stores.base import mmr_select
from rag.stores.docstore import PARENT, SQLiteDocStore
from rag.stores.filters import matches, normalize_filter, to_chroma, to_qdrant

BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param(
        "chroma",
        id="chroma",
        marks=pytest.mark.skipif(not importlib.util.find_spec("chromadb"), reason="chromadb not installed"),
    ),
    pytest.param(
        "faiss",
        id="faiss",
        marks=pytest.mark.skipif(not importlib.util.find_spec("faiss"), reason="faiss not installed"),
    ),
    pytest.param(
        "qdrant",
        id="qdrant",
        marks=pytest.mark.skipif(
            not importlib.util.find_spec("qdrant_client"), reason="qdrant not installed"
        ),
    ),
    pytest.param(
        "pgvector",
        id="pgvector",
        marks=pytest.mark.skipif(not os.environ.get("PG_CONN"), reason="PG_CONN not set"),
    ),
]

DOCS = [
    ("a" * 32, "Venus is the hottest planet in the solar system", {"source": "planets.md", "page": 1}),
    ("b" * 32, "Mars is the red planet with iron oxide dust", {"source": "planets.md", "page": 2}),
    ("c" * 32, "Espresso uses nine bars of pressure", {"source": "coffee.html", "page": 1}),
    ("d" * 32, "Pour over coffee uses a paper filter", {"source": "coffee.html", "page": 3}),
]


def docs():
    return [Document(page_content=t, metadata={**m, "chunk_id": i}, id=i) for i, t, m in DOCS]


def make(kind, tmp_path, embeddings, namespace="kw", collection="test-col", distance="cosine"):
    cfg = VectorStoreConfig.model_validate(
        {
            "type": kind,
            "collection": collection,
            "distance": distance,
            "chroma": {"persist_dir": str(tmp_path / "chroma")},
            "faiss": {"index_dir": str(tmp_path / "faiss")},
            "qdrant": {"path": str(tmp_path / "qdrant")},
        }
    )
    return build_store(cfg, embeddings, namespace=namespace, data_dir=tmp_path / "data")


@pytest.mark.parametrize("kind", BACKENDS)
def test_add_search_filter_delete(kind, tmp_path, keyword_embeddings):
    store = make(kind, tmp_path, keyword_embeddings)
    assert store.search("anything", k=3) == []
    store.add(docs())
    if store.count() is not None:
        assert store.count() == 4

    hits = store.search("hottest planet venus", k=2)
    assert hits[0].document.page_content.startswith("Venus")
    assert hits[0].document.metadata["chunk_id"] == "a" * 32
    assert hits[0].score >= hits[1].score

    filtered = store.search("planet", k=4, flt={"source": "coffee.html"})
    assert {h.document.metadata["source"] for h in filtered} == {"coffee.html"}
    ranged = store.search("coffee", k=4, flt={"page": {"$gte": 2}})
    assert {h.document.metadata["page"] for h in ranged} <= {2, 3} and ranged
    assert store.search("x", k=4, flt={"source": {"$in": ["nope.md"]}}) == []

    store.delete(["a" * 32])
    assert all(h.document.metadata["chunk_id"] != "a" * 32 for h in store.search("venus hottest", k=4))

    # upsert replaces instead of duplicating
    store.add(docs()[1:2])
    if store.count() is not None:
        assert store.count() == 3


@pytest.mark.parametrize("kind", BACKENDS)
def test_mmr_diversifies(kind, tmp_path, keyword_embeddings):
    store = make(kind, tmp_path, keyword_embeddings)
    near_dupes = [
        Document(page_content="coffee espresso pressure", metadata={"chunk_id": "1" * 32}, id="1" * 32),
        Document(page_content="coffee espresso pressure bars", metadata={"chunk_id": "2" * 32}, id="2" * 32),
        Document(page_content="coffee filter paper", metadata={"chunk_id": "3" * 32}, id="3" * 32),
    ]
    store.add(near_dupes)
    plain = [h.document.id or h.document.metadata["chunk_id"] for h in store.search("coffee espresso", k=2)]
    mmr = store.mmr_search("coffee espresso", k=2, fetch_k=3, lambda_mult=0.3)
    mmr_ids = [h.document.metadata["chunk_id"] for h in mmr]
    assert len(plain) == 2 and "3" * 32 in mmr_ids


@pytest.mark.parametrize("kind", [p for p in BACKENDS if p.id != "pgvector"])
def test_persistence_and_reset(kind, tmp_path, keyword_embeddings):
    if kind == "memory":
        pytest.skip("memory store is not persistent")
    store = make(kind, tmp_path, keyword_embeddings)
    store.add(docs())
    if kind == "qdrant":
        store._client.close()  # local qdrant allows one client per path
    reopened = make(kind, tmp_path, keyword_embeddings)
    assert reopened.count() == 4
    assert reopened.search("espresso", k=1)[0].document.page_content.startswith("Espresso")
    reopened.reset()
    assert reopened.count() == 0 and reopened.meta() == {}


@pytest.mark.parametrize("kind", ["memory", "faiss"])
def test_embedding_guard(kind, tmp_path, keyword_embeddings):
    store = make(kind, tmp_path, keyword_embeddings, namespace="google:model-a")
    store.add(docs())
    if kind == "memory":
        store.namespace = "openai:model-b"
        other = store
    else:
        other = make(kind, tmp_path, keyword_embeddings, namespace="openai:model-b")
    with pytest.raises(ConfigError, match="built with embeddings 'google:model-a'"):
        other.search("venus", k=1)
    with pytest.raises(ConfigError, match="rag store reset"):
        other.add(docs())


@pytest.mark.parametrize("distance", ["cosine", "l2", "ip"])
def test_faiss_distances(distance, tmp_path, keyword_embeddings):
    pytest.importorskip("faiss")
    store = make("faiss", tmp_path, keyword_embeddings, distance=distance)
    store.add(docs())
    assert store.search("red planet mars", k=1)[0].document.page_content.startswith("Mars")


def test_mmr_select_prefers_diverse():
    import numpy as np

    q = np.array([1.0, 0.0])
    cands = np.array([[1.0, 0.0], [0.99, 0.01], [0.7, 0.7]])
    assert mmr_select(q, cands, 2, 0.3) == [0, 2]
    assert mmr_select(q, cands, 2, 1.0) == [0, 1]
    assert mmr_select(q, np.empty((0, 2)), 2, 0.5) == []


# -------------------------------------------------------------------- filters
def test_filters():
    meta = {"source": "a.md", "page": 3}
    assert matches(meta, {"source": "a.md", "page": {"$gte": 2, "$lt": 4}})
    assert not matches(meta, {"page": {"$gt": 3}})
    assert matches(meta, {"source": {"$nin": ["b.md"]}, "missing": {"$ne": 1}})
    assert not matches(meta, {"missing": {"$gt": 1}})
    assert not matches({"page": "x"}, {"page": {"$gt": 1}})
    with pytest.raises(ConfigError, match="unknown filter operator"):
        normalize_filter({"a": {"$like": "x"}})
    with pytest.raises(ConfigError, match="needs a list"):
        normalize_filter({"a": {"$in": "x"}})


def test_filter_translation():
    assert to_chroma(None) is None
    assert to_chroma({"a": 1}) == {"a": {"$eq": 1}}
    assert to_chroma({"a": 1, "b": {"$gt": 2}}) == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}
    pytest.importorskip("qdrant_client")
    q = to_qdrant({"a": 1, "b": {"$gte": 2, "$lt": 5}, "c": {"$ne": "x"}})
    assert [c.key for c in q.must] == ["metadata.a", "metadata.b"]
    assert q.must[1].range.gte == 2 and q.must[1].range.lt == 5
    assert q.must_not[0].key == "metadata.c"


# ------------------------------------------------------------------- docstore
def test_docstore(tmp_path):
    ds = SQLiteDocStore(tmp_path / "ds.sqlite")
    v0 = ds.version()
    chunks = [
        Document(page_content=t, metadata={**m, "doc_id": m["source"], "chunk_id": i}, id=i)
        for i, t, m in DOCS
    ]
    ds.put(chunks)
    ds.put(
        [Document(page_content="parent", metadata={"doc_id": "planets.md", "chunk_id": "p1"})], kind=PARENT
    )
    assert ds.version() > v0
    assert ds.count() == 4 and ds.count(PARENT) == 1
    assert [d.id for d in ds.get_many(["c" * 32, "zz", "a" * 32])] == ["c" * 32, "a" * 32]
    assert len(ds.all(flt={"source": "coffee.html"})) == 2
    assert ds.sources() == {"planets.md": 2, "coffee.html": 2}
    deleted = ds.delete_doc("planets.md")
    assert sorted(deleted) == ["a" * 32, "b" * 32]
    assert ds.count(PARENT) == 0 and ds.count() == 2
    ds.reset()
    assert ds.count() == 0
