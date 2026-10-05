from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.config.schema import CleaningConfig, IngestionConfig, SplitterConfig
from rag.core.exceptions import ConfigError, RAGError
from rag.ingestion.cleaners import clean_documents, normalize_text, strip_repeated_lines
from rag.ingestion.loaders import discover_sources, html_to_text, load_source, select_path
from rag.ingestion.manifest import Manifest
from rag.ingestion.metadata import enrich_chunks, normalize_source, sanitize_metadata
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.splitters import build_splitter
from rag.stores import build_store
from rag.stores.docstore import PARENT, SQLiteDocStore
from tests.conftest import FIXTURES, make_pdf

ING = IngestionConfig()


# -------------------------------------------------------------------- loaders
def test_load_markdown_and_text():
    docs = load_source(str(FIXTURES / "sample.md"), ING)
    assert len(docs) == 1 and "Venus" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "md"


def test_load_pdf_pages(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", ["First page about apples", "Second page about pears"])
    docs = load_source(str(pdf), ING)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "pears" in docs[1].page_content


def test_load_html_strips_scripts_and_keeps_headings():
    docs = load_source(str(FIXTURES / "sample.html"), ING)
    text = docs[0].page_content
    assert docs[0].metadata["title"] == "Coffee Brewing"
    assert "# Coffee Brewing" in text and "## Espresso" in text
    assert "var x" not in text and "color:red" not in text
    assert "- Use fresh beans" in text


def test_load_csv_rows_and_metadata_columns():
    cfg = IngestionConfig.model_validate(
        {"csv": {"content_columns": ["product", "description"], "metadata_columns": ["category"]}}
    )
    docs = load_source(str(FIXTURES / "sample.csv"), cfg)
    assert len(docs) == 3
    assert docs[0].page_content.startswith("product: Trail Runner")
    assert "price" not in docs[0].page_content
    assert docs[1].metadata["category"] == "jackets" and docs[1].metadata["row"] == 2


def test_load_json_path_and_content_key():
    cfg = IngestionConfig.model_validate({"json": {"jq_schema": ".faq[]", "content_key": "answer"}})
    docs = load_source(str(FIXTURES / "sample.json"), cfg)
    assert [d.metadata["topic"] for d in docs] == ["shipping", "returns"]
    assert docs[0].page_content.startswith("Standard shipping")


def test_load_json_whole_document():
    docs = load_source(str(FIXTURES / "sample.json"), ING)
    assert len(docs) == 1 and "faq" in docs[0].page_content


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "alpha"}\n\n{"text": "beta"}\n')
    cfg = IngestionConfig.model_validate({"json": {"content_key": "text"}})
    assert [d.page_content for d in load_source(str(path), cfg)] == ["alpha", "beta"]


def test_select_path():
    data = {"a": {"items": [{"t": 1}, {"t": 2}]}}
    assert select_path(data, ".a.items[]") == [{"t": 1}, {"t": 2}]
    assert select_path(data, ".a.items[].t") == [1, 2]
    assert select_path([1, 2], ".[]") == [1, 2]
    with pytest.raises(RAGError, match="not a list"):
        select_path(data, ".a[]")


def test_loader_override_and_unsupported(tmp_path):
    path = tmp_path / "notes.weird"
    path.write_text("plain text content")
    with pytest.raises(RAGError, match="no loader"):
        load_source(str(path), ING)
    cfg = IngestionConfig.model_validate({"loaders": {".WEIRD": "text"}})
    assert load_source(str(path), cfg)[0].page_content == "plain text content"


def test_discover_sources(tmp_path, corpus):
    (corpus / "image.png").write_bytes(b"x")
    (corpus / "skip").mkdir()
    (corpus / "skip" / "a.md").write_text("# hidden")
    found = discover_sources(
        [str(corpus), "https://example.com/x", str(corpus / "sample.md")], exclude=["skip/*"]
    )
    names = [Path(f).name for f in found if not f.startswith("http")]
    assert sorted(names) == ["sample.csv", "sample.html", "sample.json", "sample.md", "sample.txt"]
    assert "https://example.com/x" in found
    assert len(found) == len(set(found))


def test_html_to_text_links():
    text, title, links = html_to_text('<title>T</title><a href="/x">x</a><h3>Deep</h3>')
    assert title == "T" and links == ["/x"] and "### Deep" in text


# ------------------------------------------------------------------- cleaners
def test_normalize_text():
    assert normalize_text("a  \t b  \r\n\r\n\r\n\nc ") == "a b\n\nc"


def test_strip_repeated_lines():
    pages = [
        Document(
            page_content=f"ACME Corp Confidential\nBody {i} text\nPage {i} of 4",
            metadata={"source": "x.pdf", "page": i},
        )
        for i in range(1, 5)
    ]
    out = strip_repeated_lines(pages)
    assert all("ACME" not in d.page_content and "Page" not in d.page_content for d in out)
    assert out[2].page_content == "Body 3 text"


def test_clean_documents_drops_empty():
    docs = [Document(page_content="  "), Document(page_content="x  y")]
    out = clean_documents(docs, CleaningConfig())
    assert [d.page_content for d in out] == ["x y"]


# ------------------------------------------------------------------ splitters
def test_recursive_splitter_respects_size():
    splitter = build_splitter(SplitterConfig(chunk_size=80, chunk_overlap=10))
    chunks = splitter.split_documents([Document(page_content="word " * 100, metadata={"source": "s"})])
    assert len(chunks) > 3 and all(len(c.page_content) <= 80 for c in chunks)
    assert all(c.metadata["source"] == "s" for c in chunks)


def test_markdown_header_splitter_sections():
    splitter = build_splitter(SplitterConfig(type="markdown_header", chunk_size=400, chunk_overlap=0))
    docs = load_source(str(FIXTURES / "sample.md"), ING)
    chunks = splitter.split_documents(docs)
    sections = {c.metadata.get("section") for c in chunks}
    assert "Solar System Guide > Inner Planets" in sections
    assert "Solar System Guide > Outer Planets" in sections


def test_html_header_uses_markdown_headings():
    splitter = build_splitter(SplitterConfig(type="html_header", chunk_size=400, chunk_overlap=0))
    chunks = splitter.split_documents(load_source(str(FIXTURES / "sample.html"), ING))
    assert any(c.metadata.get("section") == "Coffee Brewing > Espresso" for c in chunks)


def test_semantic_splitter_requires_embeddings():
    with pytest.raises(ConfigError, match="embeddings"):
        build_splitter(SplitterConfig(type="semantic"))


# ------------------------------------------------------------------- metadata
def test_normalize_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d").mkdir()
    assert normalize_source("./d/../d/a.md") == "d/a.md"
    assert normalize_source("https://x.org/a") == "https://x.org/a"
    assert normalize_source("/elsewhere/a.md").startswith("/")


def test_enrich_chunks_stable_ids():
    chunks = [Document(page_content="alpha", metadata={"page": 1, "skip": None, "tags": ["a"]})]
    a = enrich_chunks(chunks, doc_id="d1", content_hash="h", extra={"team": "x"})
    b = enrich_chunks(chunks, doc_id="d1", content_hash="h")
    assert a[0].id == b[0].id == a[0].metadata["chunk_id"] and len(a[0].id) == 32
    assert "skip" not in a[0].metadata and a[0].metadata["tags"] == '["a"]'
    assert a[0].metadata["team"] == "x"


def test_sanitize_metadata():
    assert sanitize_metadata({"a": 1, "b": None, "c": {"x": 1}}) == {"a": 1, "c": '{"x": 1}'}


# ------------------------------------------------------------------- pipeline
@pytest.fixture
def pipeline_factory(make_cfg, keyword_embeddings, tmp_path):
    def build(*overrides):
        cfg = make_cfg("ingestion.cleaning.min_chars=10", *overrides)
        store = build_store(cfg.vector_store, keyword_embeddings, namespace="kw", data_dir=cfg.app.data_dir)
        docstore = SQLiteDocStore(tmp_path / "docstore.sqlite")
        return IngestionPipeline(cfg, store, docstore, Manifest(tmp_path / "manifest.json"))

    return build


def test_pipeline_ingests_corpus(pipeline_factory, corpus, monkeypatch):
    monkeypatch.chdir(corpus.parent)
    pipe = pipeline_factory()
    report = pipe.run([str(corpus)])
    assert report.ok, report.errors
    assert report.files_seen == report.files_ingested == 5
    assert report.chunks_added == pipe.store.count() == pipe.docstore.count()
    sources = pipe.docstore.sources()
    assert "corpus/sample.md" in sources
    hit = pipe.store.search("hottest planet Venus", k=1)[0]
    assert hit.document.metadata["source"] == "corpus/sample.md"
    assert hit.document.metadata["embedding_model"] == "kw"


def test_pipeline_incremental_skip_and_update(pipeline_factory, corpus):
    pipe = pipeline_factory()
    first = pipe.run([str(corpus)])
    again = pipe.run([str(corpus)])
    assert again.files_skipped == 5 and again.chunks_added == 0
    total = pipe.store.count()

    (corpus / "sample.txt").write_text("Completely new content about volcanoes and lava flows erupting.")
    changed = pipe.run([str(corpus)])
    assert changed.files_ingested == 1 and changed.files_skipped == 4
    assert changed.chunks_deleted >= 1
    assert pipe.store.count() == total - changed.chunks_deleted + changed.chunks_added
    assert "volcanoes" in pipe.store.search("lava volcanoes", k=1)[0].document.page_content
    forced = pipe.run([str(corpus)], incremental=False)
    assert forced.files_ingested == 5 and pipe.store.count() == pipe.docstore.count()
    assert first.chunks_added > 0


def test_pipeline_records_errors_and_continues(pipeline_factory, corpus, tmp_path):
    bad = tmp_path / "empty.txt"
    bad.write_text("   ")
    report = pipeline_factory().run([str(corpus / "sample.md"), str(bad), str(tmp_path / "missing.md")])
    assert report.files_ingested == 1
    assert len(report.errors) == 2
    assert any("no text" in e for e in report.errors.values())
    assert any("FileNotFoundError" in e for e in report.errors.values())


def test_pipeline_delete_source_and_reset(pipeline_factory, corpus):
    pipe = pipeline_factory()
    pipe.run([str(corpus)])
    md = next(s for s in pipe.docstore.sources() if s.endswith("sample.md"))
    deleted = pipe.delete_source(md)
    assert deleted > 0 and md not in pipe.docstore.sources()
    assert pipe.manifest.get(md) is None
    pipe.reset()
    assert pipe.store.count() == pipe.docstore.count() == 0


def test_pipeline_tags(pipeline_factory, corpus):
    pipe = pipeline_factory()
    pipe.run([str(corpus / "sample.txt")], tags={"team": "bio"})
    assert all(d.metadata["team"] == "bio" for d in pipe.docstore.all())


def test_pipeline_parent_mode(pipeline_factory, corpus):
    pipe = pipeline_factory(
        "retrieval.strategy=parent",
        "retrieval.parent.parent_chunk_size=300",
        "retrieval.parent.child_chunk_size=100",
    )
    pipe.run([str(corpus / "sample.md")])
    parents = pipe.docstore.all(kind=PARENT)
    children = pipe.docstore.all()
    assert parents and len(children) > len(parents)
    parent_ids = {p.metadata["chunk_id"] for p in parents}
    assert all(c.metadata["parent_id"] in parent_ids for c in children)
    assert all(len(c.page_content) <= 100 for c in children)
    assert pipe.store.count() == len(children)


def test_manifest_roundtrip(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.set("a", content_hash="h1", doc_id="d", chunk_ids=["c"])
    m.save()
    again = Manifest(tmp_path / "m.json")
    assert again.is_current("a", "h1") and not again.is_current("a", "h2")
    memory_only = Manifest(None)
    memory_only.set("a", content_hash="h")
    memory_only.save()  # no-op
