"""Generate an evaluation dataset from the documents already in the index.

Questions are written by the configured LLM from sampled chunks, then validated:
evidence quotes must appear verbatim in the source chunk, questions must be
non-trivial and not near-duplicates. Generated items are marked
``generated: true`` and should be reviewed before being used as a benchmark.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag_chatbot.core.exceptions import RAGError
from rag_chatbot.core.logging import get_logger
from rag_chatbot.evaluation.dataset import CATEGORIES, QAItem
from rag_chatbot.evaluation.metrics.retrieval import contains_evidence
from rag_chatbot.generation import prompts
from rag_chatbot.generation.schemas import GeneratedFollowUp, GeneratedQA, GeneratedQuestion
from rag_chatbot.pipeline import RAGPipeline

log = get_logger("evaluation.generate")

GENERATABLE = ("factual", "multi_hop", "unanswerable", "conversational")
DEFAULT_MIX = {"factual": 0.5, "multi_hop": 0.2, "unanswerable": 0.15, "conversational": 0.15}
MAX_EXCERPT_CHARS = 2500
_WORD = re.compile(r"\w+")


@dataclass
class GenerationResult:
    items: list[QAItem] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # one reason per rejected attempt


def allocate(n: int, categories: Sequence[str]) -> dict[str, int]:
    """Split ``n`` across categories following DEFAULT_MIX (largest remainders)."""
    unknown = set(categories) - set(GENERATABLE)
    if unknown:
        raise RAGError(f"cannot generate {sorted(unknown)}; choose from {', '.join(GENERATABLE)}")
    weights = {c: DEFAULT_MIX[c] for c in categories}
    total = sum(weights.values())
    raw = {c: n * w / total for c, w in weights.items()}
    counts = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - counts[c], reverse=True)[: n - sum(counts.values())]:
        counts[c] += 1
    return counts


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def is_near_duplicate(question: str, existing: Sequence[str], threshold: float = 0.8) -> bool:
    words = _words(question)
    for other in existing:
        theirs = _words(other)
        union = words | theirs
        if union and len(words & theirs) / len(union) >= threshold:
            return True
    return False


class DatasetGenerator:
    def __init__(self, pipeline: RAGPipeline, llm: BaseChatModel | None = None, seed: int = 0) -> None:
        self.pipeline = pipeline
        self._llm = llm
        self.rng = random.Random(seed)
        chunks = pipeline.docstore.all()
        if not chunks:
            raise RAGError("the index is empty; run `rag ingest` first")
        self.by_source: dict[str, list[Document]] = defaultdict(list)
        for chunk in chunks:
            if len(chunk.page_content.strip()) >= 80:
                self.by_source[str(chunk.metadata.get("source", "?"))].append(chunk)
        if not self.by_source:
            raise RAGError("no chunks long enough to generate questions from")
        self._cursor = 0

    @property
    def llm(self) -> BaseChatModel:
        return self._llm or self.pipeline.llm

    # ------------------------------------------------------------- sampling
    def _sample(self) -> Document:
        """Round-robin over sources so every document is represented."""
        sources = sorted(self.by_source)
        source = sources[self._cursor % len(sources)]
        self._cursor += 1
        return self.rng.choice(self.by_source[source])

    def _sample_pair(self) -> tuple[Document, Document]:
        sources = sorted(self.by_source)
        if len(sources) < 2:
            raise RAGError("multi-hop questions need at least two indexed sources")
        a, b = self.rng.sample(sources, 2)
        return self.rng.choice(self.by_source[a]), self.rng.choice(self.by_source[b])

    @staticmethod
    def _name(doc: Document) -> str:
        return Path(str(doc.metadata.get("source", "?"))).name

    @staticmethod
    def _text(doc: Document) -> str:
        return doc.page_content[:MAX_EXCERPT_CHARS]

    # ------------------------------------------------------------ builders
    def _factual(self, n: int) -> QAItem:
        doc = self._sample()
        qa: GeneratedQA = (prompts.GENERATE_QA | self.llm.with_structured_output(GeneratedQA)).invoke(  # type: ignore[assignment]
            {"context": self._text(doc)}
        )
        evidence = [e.strip() for e in qa.evidence if e.strip()]
        if not evidence or not all(contains_evidence(doc.page_content, [e]) for e in evidence):
            raise ValueError("evidence is not a verbatim quote from the chunk")
        return QAItem(
            id=f"gen-factual-{n}",
            category="factual",
            question=qa.question.strip(),
            ground_truth=qa.answer.strip(),
            expected_sources=[self._name(doc)],
            evidence=evidence,
            generated=True,
        )

    def _multi_hop(self, n: int) -> QAItem:
        a, b = self._sample_pair()
        chain = prompts.GENERATE_MULTIHOP | self.llm.with_structured_output(GeneratedQA)
        qa: GeneratedQA = chain.invoke({"context_a": self._text(a), "context_b": self._text(b)})  # type: ignore[assignment]
        evidence = [e.strip() for e in qa.evidence if e.strip()]
        if len(evidence) < 2:
            raise ValueError("multi-hop items need a quote from each excerpt")
        if not contains_evidence(a.page_content, [evidence[0]]) or not contains_evidence(
            b.page_content, [evidence[1]]
        ):
            raise ValueError("multi-hop evidence is not quoted verbatim from both excerpts")
        return QAItem(
            id=f"gen-multi_hop-{n}",
            category="multi_hop",
            question=qa.question.strip(),
            ground_truth=qa.answer.strip(),
            expected_sources=sorted({self._name(a), self._name(b)}),
            evidence=evidence[:2],
            generated=True,
        )

    def _unanswerable(self, n: int) -> QAItem:
        docs = [self._sample() for _ in range(3)]
        context = "\n\n".join(self._text(d) for d in docs)
        chain = prompts.GENERATE_UNANSWERABLE | self.llm.with_structured_output(GeneratedQuestion)
        q: GeneratedQuestion = chain.invoke({"context": context})  # type: ignore[assignment]
        return QAItem(
            id=f"gen-unanswerable-{n}", category="unanswerable", question=q.question.strip(), generated=True
        )

    def _conversational(self, n: int) -> QAItem:
        base = self._factual(n)
        chain = prompts.GENERATE_FOLLOWUP | self.llm.with_structured_output(GeneratedFollowUp)
        conv: GeneratedFollowUp = chain.invoke({"question": base.question})  # type: ignore[assignment]
        if not conv.first_question.strip() or not conv.follow_up.strip():
            raise ValueError("empty conversation turn")
        return base.model_copy(
            update={
                "id": f"gen-conversational-{n}",
                "category": "conversational",
                "question": conv.follow_up.strip(),
                "history": [conv.first_question.strip()],
            }
        )

    # ---------------------------------------------------------------- run
    def generate(
        self, n: int, categories: Sequence[str] = GENERATABLE, max_attempts: int = 3
    ) -> GenerationResult:
        builders = {
            "factual": self._factual,
            "multi_hop": self._multi_hop,
            "unanswerable": self._unanswerable,
            "conversational": self._conversational,
        }
        result = GenerationResult()
        for category, count in allocate(n, categories).items():
            made, attempts = 0, 0
            while made < count and attempts < count * max_attempts:
                attempts += 1
                try:
                    item = builders[category](made + 1)
                    if len(item.question) < 10:
                        raise ValueError("question too short")
                    if is_near_duplicate(item.question, [i.question for i in result.items]):
                        raise ValueError("near-duplicate question")
                except Exception as exc:  # rejected attempt; try another sample
                    result.rejected.append(f"{category}: {exc}")
                    log.debug("rejected %s question: %s", category, exc)
                    continue
                result.items.append(item)
                made += 1
            if made < count:
                log.warning("generated %d/%d %s questions", made, count, category)
        return result


__all__ = [
    "CATEGORIES",
    "DEFAULT_MIX",
    "DatasetGenerator",
    "GenerationResult",
    "allocate",
    "is_near_duplicate",
]
