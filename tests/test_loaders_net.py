"""URL and DOCX loaders against real (local) inputs."""

import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag_chatbot.config.schema import IngestionConfig
from rag_chatbot.core.exceptions import RAGError
from rag_chatbot.ingestion import loaders
from rag_chatbot.ingestion.loaders import load_source
from tests.conftest import make_pdf


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Serve a tiny site: index -> page2, a PDF, an image and an external link."""
    root = tmp_path / "site"
    root.mkdir()
    (root / "index.html").write_text(
        "<html><head><title>Home</title></head><body><h1>Home</h1><p>Welcome to the docs.</p>"
        '<a href="page2.html">next</a> <a href="manual.pdf">manual</a> <a href="logo.png">logo</a>'
        '<a href="http://example.invalid/elsewhere">external</a></body></html>'
    )
    (root / "page2.html").write_text(
        '<html><body><h2>Second</h2><p>Deeper content.</p><a href="page3.html">more</a></body></html>'
    )
    (root / "page3.html").write_text("<html><body><p>Too deep.</p></body></html>")
    make_pdf(root / "manual.pdf", ["Manual page one about setup", "Manual page two about usage"])
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    (root / "notes.txt").write_text("plain text notes")

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def url_cfg(recursive=False, depth=2):
    return IngestionConfig.model_validate({"url": {"recursive": recursive, "max_depth": depth}})


def test_single_page(site):
    docs = load_source(f"{site}/index.html", url_cfg())
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Home" and "# Home" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "url"


def test_recursive_crawl_same_host_with_pdf(site):
    docs = load_source(f"{site}/index.html", url_cfg(recursive=True, depth=2))
    sources = [d.metadata["source"] for d in docs]
    assert f"{site}/page2.html" in sources
    assert f"{site}/page3.html" not in sources  # beyond max_depth
    assert not any("example.invalid" in s for s in sources)  # other host
    assert not any(s.endswith("logo.png") for s in sources)  # binary skipped
    pdf = [d for d in docs if d.metadata["source"].endswith("manual.pdf")]
    assert [d.metadata["page"] for d in pdf] == [1, 2]
    assert "usage" in pdf[1].page_content


def test_direct_pdf_and_text_urls(site):
    pdf = load_source(f"{site}/manual.pdf", url_cfg())
    assert [d.metadata["page"] for d in pdf] == [1, 2]
    text = load_source(f"{site}/notes.txt", url_cfg())
    assert text[0].page_content == "plain text notes"


def test_size_cap(site, monkeypatch):
    monkeypatch.setattr(loaders, "MAX_URL_BYTES", 50)
    with pytest.raises(RAGError, match="larger than"):
        load_source(f"{site}/index.html", url_cfg())


def test_docx_loader(tmp_path):
    pytest.importorskip("docx2txt")
    path = tmp_path / "report.docx"
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        "<w:p><w:r><w:t>Quarterly revenue grew by twelve percent.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Costs stayed flat.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types"><Default Extension="xml" ContentType="application/xml"/></Types>',
        )
        z.writestr("word/document.xml", body)
    docs = load_source(str(path), IngestionConfig())
    assert "twelve percent" in docs[0].page_content and "Costs stayed flat." in docs[0].page_content
    assert docs[0].metadata["file_type"] == "docx"
