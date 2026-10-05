"""Text cleanup applied to loaded documents before splitting."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from langchain_core.documents import Document

from rag.config.schema import CleaningConfig

_SPACES = re.compile(r"[ \t\f\v ]+")
_BLANKS = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _numberless(line: str) -> str:
    # "Page 3 of 10" and "Page 4 of 10" are the same footer
    return "#num:" + _DIGITS.sub("#", line.strip().lower())


def _edge_keys(lines: list[str], edge: int) -> list[tuple[int, str]]:
    """Candidate header/footer keys: exact text of the first/last ``edge`` lines,
    plus a digit-insensitive key for the very first and very last line."""
    idx = [i for i, ln in enumerate(lines) if 0 < len(ln.strip()) < 100]
    if not idx:
        return []
    edges = set(idx[:edge] + idx[-edge:])
    keys = [(i, lines[i].strip().lower()) for i in sorted(edges)]
    keys += [(i, _numberless(lines[i])) for i in {idx[0], idx[-1]}]
    return keys


def strip_repeated_lines(
    docs: list[Document], min_pages: int = 3, ratio: float = 0.5, edge: int = 3
) -> list[Document]:
    """Remove header/footer lines repeated on >= ``ratio`` of a source's pages.

    Only the first/last ``edge`` lines of a page are considered, so body text
    is never removed.
    """
    by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        by_source[str(doc.metadata.get("source"))].append(doc)
    out: list[Document] = []
    for pages in by_source.values():
        if len(pages) < min_pages:
            out.extend(pages)
            continue
        page_lines = [p.page_content.split("\n") for p in pages]
        page_keys = [_edge_keys(lines, edge) for lines in page_lines]
        counts: Counter[str] = Counter()
        for keys in page_keys:
            counts.update({k for _, k in keys})
        repeated = {k for k, n in counts.items() if n >= max(2, ratio * len(pages))}
        for page, lines, keys in zip(pages, page_lines, page_keys, strict=True):
            drop = {i for i, k in keys if k in repeated}
            kept = [ln for i, ln in enumerate(lines) if i not in drop]
            out.append(Document(page_content="\n".join(kept), metadata=page.metadata, id=page.id))
    return out


def clean_documents(docs: list[Document], cfg: CleaningConfig) -> list[Document]:
    if cfg.strip_headers_footers:
        docs = strip_repeated_lines(docs)
    if cfg.normalize_whitespace:
        docs = [
            Document(page_content=normalize_text(d.page_content), metadata=d.metadata, id=d.id) for d in docs
        ]
    return [d for d in docs if d.page_content.strip()]


def drop_short(chunks: Iterable[Document], min_chars: int) -> list[Document]:
    return [c for c in chunks if len(c.page_content.strip()) >= min_chars]
