"""Document loaders: a source path or URL -> list of LangChain ``Document``.

The loader is picked by file extension (``DEFAULT_LOADERS``), and
``ingestion.loaders`` can remap any extension to another registered loader.
HTML is converted to text with markdown-style ``#`` headings, so the header
-aware splitters work for both Markdown and HTML.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import urllib.request
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from langchain_core.documents import Document

from rag_chatbot.config.schema import IngestionConfig
from rag_chatbot.core.exceptions import RAGError
from rag_chatbot.core.registry import Registry, require

Loader = Callable[[str, IngestionConfig], list[Document]]
LOADERS: Registry[Loader] = Registry("loader")

DEFAULT_LOADERS = {
    "pdf": "pypdf",
    "txt": "text",
    "text": "text",
    "md": "text",
    "markdown": "text",
    "rst": "text",
    "log": "text",
    "docx": "docx",
    "html": "bs4",
    "htm": "bs4",
    "csv": "csv",
    "json": "json",
    "jsonl": "jsonl",
}

USER_AGENT = "ragchatbot/0.1 (+https://github.com/msmidhunms/RAGChatbot)"
MAX_URL_PAGES = 50


class UnsupportedSourceError(RAGError):
    pass


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def extension(source: str) -> str:
    return Path(source).suffix.lower().lstrip(".")


def resolve_loader(source: str, cfg: IngestionConfig) -> str:
    if is_url(source):
        return cfg.loaders.get("url", "url")
    ext = extension(source)
    name = cfg.loaders.get(ext) or DEFAULT_LOADERS.get(ext)
    if not name:
        raise UnsupportedSourceError(f"no loader for '.{ext}' files ({source}); map one in ingestion.loaders")
    return name


def is_supported(source: str, cfg: IngestionConfig) -> bool:
    ext = extension(source)
    return ext in cfg.loaders or ext in DEFAULT_LOADERS


def load_source(source: str, cfg: IngestionConfig) -> list[Document]:
    docs = LOADERS.get(resolve_loader(source, cfg))(source, cfg)
    file_type = "url" if is_url(source) else extension(source)
    for doc in docs:
        doc.metadata.setdefault("source", source)
        doc.metadata.setdefault("file_type", file_type)
    return [d for d in docs if d.page_content.strip()]


def discover_sources(
    sources: Iterable[str],
    glob: str = "**/*",
    exclude: Iterable[str] = (),
    *,
    cfg: IngestionConfig | None = None,
) -> list[str]:
    """Expand directories into supported files; files and URLs pass through.

    Missing paths are kept so the pipeline can report them as errors.
    """
    cfg = cfg or IngestionConfig()
    patterns = list(exclude)
    out: list[str] = []
    for src in sources:
        if is_url(src):
            out.append(src)
            continue
        path = Path(src)
        if path.is_dir():
            for p in sorted(path.glob(glob)):
                rel = p.relative_to(path).as_posix()
                if not p.is_file() or not is_supported(p.name, cfg):
                    continue
                if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns):
                    continue
                out.append(str(p))
        else:
            out.append(src)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ helpers
def read_text(path: str) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover - latin-1 never fails


class _HTMLToText(HTMLParser):
    """Minimal HTML -> text with markdown headings; no external dependency."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "tr", "section", "article", "header", "footer", "pre", "blockquote", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.links: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self._skip += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") or tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(html: str) -> tuple[str, str, list[str]]:
    """Return (text, title, links)."""
    parser = _HTMLToText()
    parser.feed(html)
    parser.close()
    return parser.text(), parser.title.strip(), parser.links


# ------------------------------------------------------------------ loaders
@LOADERS.register("text")
def load_text(source: str, cfg: IngestionConfig) -> list[Document]:
    return [Document(page_content=read_text(source), metadata={"source": source})]


@LOADERS.register("pypdf")
def load_pypdf(source: str, cfg: IngestionConfig) -> list[Document]:
    pypdf = require("pypdf")
    reader = pypdf.PdfReader(source)
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": source, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


@LOADERS.register("pymupdf")
def load_pymupdf(source: str, cfg: IngestionConfig) -> list[Document]:
    pymupdf = require("pymupdf", "docs")
    with pymupdf.open(source) as pdf:
        return [
            Document(page_content=page.get_text(), metadata={"source": source, "page": i})
            for i, page in enumerate(pdf, start=1)
        ]


@LOADERS.register("docx")
def load_docx(source: str, cfg: IngestionConfig) -> list[Document]:
    docx2txt = require("docx2txt", "docs")
    return [Document(page_content=docx2txt.process(source) or "", metadata={"source": source})]


@LOADERS.register("html")
@LOADERS.register("bs4")
def load_html(source: str, cfg: IngestionConfig) -> list[Document]:
    text, title, _ = html_to_text(read_text(source))
    meta: dict[str, Any] = {"source": source}
    if title:
        meta["title"] = title
    return [Document(page_content=text, metadata=meta)]


def _fetch(url: str, timeout: float = 20.0) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user supplied source
        content_type = resp.headers.get("Content-Type", "")
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace"), content_type


@LOADERS.register("url")
def load_url(source: str, cfg: IngestionConfig) -> list[Document]:
    host = urlparse(source).netloc
    max_depth = cfg.url.max_depth if cfg.url.recursive else 1
    queue: list[tuple[str, int]] = [(source, 1)]
    seen: set[str] = set()
    docs: list[Document] = []
    while queue and len(seen) < MAX_URL_PAGES:
        url, depth = queue.pop(0)
        url = url.split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        body, content_type = _fetch(url)
        if "html" in content_type or body.lstrip().lower().startswith(("<!doctype html", "<html")):
            text, title, links = html_to_text(body)
            meta: dict[str, Any] = {"source": url}
            if title:
                meta["title"] = title
            docs.append(Document(page_content=text, metadata=meta))
            if depth < max_depth:
                for link in links:
                    nxt = urljoin(url, link)
                    if urlparse(nxt).netloc == host and nxt.startswith(("http://", "https://")):
                        queue.append((nxt, depth + 1))
        else:
            docs.append(Document(page_content=body, metadata={"source": url}))
    return docs


@LOADERS.register("csv")
def load_csv(source: str, cfg: IngestionConfig) -> list[Document]:
    content_cols = cfg.csv.content_columns
    meta_cols = cfg.csv.metadata_columns
    docs = []
    with open(source, newline="", encoding="utf-8-sig") as fh:
        for row_num, row in enumerate(csv.DictReader(fh), start=1):
            cols = content_cols or [c for c in row if c not in meta_cols]
            content = "\n".join(f"{c}: {row.get(c, '')}" for c in cols)
            meta: dict[str, Any] = {"source": source, "row": row_num}
            meta.update({c: row.get(c) for c in meta_cols})
            docs.append(Document(page_content=content, metadata=meta))
    return docs


def select_path(data: Any, schema: str) -> list[Any]:
    """Tiny jq subset: ``.``, ``.a.b``, ``.items[]``, ``.items[].text``."""
    items = [data]
    for token in [t for t in schema.strip().split(".") if t]:
        iterate = token.endswith("[]")
        key = token[:-2] if iterate else token
        nxt: list[Any] = []
        for item in items:
            value = item.get(key) if key and isinstance(item, dict) else (item if not key else None)
            if value is None:
                continue
            if iterate:
                if not isinstance(value, list):
                    raise RAGError(f"json path {schema!r}: '{key or '.'}' is not a list")
                nxt.extend(value)
            else:
                nxt.append(value)
        items = nxt
    return items


def _json_docs(source: str, records: list[Any], cfg: IngestionConfig) -> list[Document]:
    key = cfg.json_.content_key
    docs = []
    for seq, item in enumerate(records, start=1):
        if key is not None:
            if not isinstance(item, dict) or key not in item:
                continue
            value = item[key]
        else:
            value = item
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
        meta: dict[str, Any] = {"source": source, "seq_num": seq}
        if isinstance(item, dict):
            meta.update(
                {k: v for k, v in item.items() if k != key and isinstance(v, (str, int, float, bool))}
            )
        docs.append(Document(page_content=content, metadata=meta))
    return docs


@LOADERS.register("json")
def load_json(source: str, cfg: IngestionConfig) -> list[Document]:
    if cfg.json_.json_lines:
        return load_jsonl(source, cfg)
    data = json.loads(read_text(source))
    return _json_docs(source, select_path(data, cfg.json_.jq_schema), cfg)


@LOADERS.register("jsonl")
def load_jsonl(source: str, cfg: IngestionConfig) -> list[Document]:
    records: list[Any] = []
    for line in read_text(source).splitlines():
        if line.strip():
            records.extend(select_path(json.loads(line), cfg.json_.jq_schema))
    return _json_docs(source, records, cfg)
