"""Answer generation from retrieved context.

Structured mode asks the model for ``LLMAnswer`` (falling back to plain
text if the provider cannot do structured output); text mode, also used
for streaming, parses ``[n]`` citation markers from the text. Citations are
always resolved against the chunks actually placed in the prompt.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from langchain_core.output_parsers import StrOutputParser

from rag.config.schema import RAGConfig
from rag.core.logging import get_logger
from rag.core.types import RetrievedChunk
from rag.generation.context import format_context, short_label
from rag.generation.prompts import answer_prompt
from rag.generation.schemas import Citation, LLMAnswer, RAGAnswer
from rag.retrieval.base import LLMGetter

log = get_logger("generation")

NO_ANSWER = "I couldn't find information about that in the indexed documents."
NO_CONTEXT = "(no relevant documents were found)"
ANSWER_TAG = "rag-answer"  # tags the LLM call whose tokens are streamed to the user
_CITE = re.compile(r"\[(\d+)\]")


def history_section(history: str) -> str:
    return f"Conversation so far:\n{history}\n\n" if history.strip() else ""


def feedback_section(feedback: str) -> str:
    if not feedback:
        return ""
    return (
        f"\n\nA previous draft of the answer contained claims not supported by the context: {feedback}\n"
        "Write the answer again using only information from the context."
    )


class Generator:
    def __init__(self, get_llm: LLMGetter, cfg: RAGConfig) -> None:
        self._get_llm = get_llm
        self.cfg = cfg
        self.gen = cfg.generation
        self.prompt = answer_prompt(cfg.generation)

    # ------------------------------------------------------------------ api
    def refuses(self, chunks: Sequence[RetrievedChunk]) -> bool:
        return not chunks and self.gen.answer_when_no_context == "refuse"

    def no_answer(self) -> RAGAnswer:
        return RAGAnswer(title="No answer found", answer=NO_ANSWER, citations=[], confidence="low")

    def prepare(
        self, question: str, chunks: Sequence[RetrievedChunk], history: str = "", feedback: str = ""
    ) -> tuple[dict[str, str], list[RetrievedChunk]]:
        context, used = format_context(chunks, self.gen.max_context_tokens)
        values = {
            "question": question,
            "context": context or NO_CONTEXT,
            "history": history_section(history),
            "feedback": feedback_section(feedback),
        }
        return values, used

    def generate(
        self, question: str, chunks: Sequence[RetrievedChunk], history: str = "", feedback: str = ""
    ) -> RAGAnswer:
        if self.refuses(chunks):
            return self.no_answer()
        values, used = self.prepare(question, chunks, history, feedback)
        if self.cfg.llm.structured_output:
            try:
                chain = self.prompt | self._get_llm().with_structured_output(LLMAnswer)
                result = chain.invoke(values)
                if isinstance(result, LLMAnswer):
                    return self.from_structured(result, used)
                log.warning("structured output returned %s; falling back to text", type(result).__name__)
            except NotImplementedError:
                log.info("provider has no structured output; using text mode")
            except Exception as exc:  # malformed JSON, schema violations, ...
                log.warning("structured generation failed (%s); falling back to text", exc)
        text = (self.prompt | self._get_llm() | StrOutputParser()).invoke(values)
        return self.from_text(text, used)

    def stream(
        self, question: str, chunks: Sequence[RetrievedChunk], history: str = "", feedback: str = ""
    ) -> Iterator[str]:
        """Yield answer tokens (text mode). Use ``from_text`` on the joined tokens afterwards."""
        if self.refuses(chunks):
            yield NO_ANSWER
            return
        values, _ = self.prepare(question, chunks, history, feedback)
        llm = self._get_llm().with_config(tags=[ANSWER_TAG])
        yield from (self.prompt | llm | StrOutputParser()).stream(values)

    def generate_text(
        self, question: str, chunks: Sequence[RetrievedChunk], history: str = "", feedback: str = ""
    ) -> RAGAnswer:
        """Text-mode generation tagged for token streaming through the graph."""
        if self.refuses(chunks):
            return self.no_answer()
        values, used = self.prepare(question, chunks, history, feedback)
        llm = self._get_llm().with_config(tags=[ANSWER_TAG])
        return self.from_text((self.prompt | llm | StrOutputParser()).invoke(values), used)

    # ------------------------------------------------------------ assembly
    def _citations(self, ids: Sequence[int], used: Sequence[RetrievedChunk]) -> list[Citation]:
        if self.gen.citation_style == "none":
            return []
        out = []
        for i in dict.fromkeys(ids):
            if 1 <= i <= len(used):
                meta = used[i - 1].metadata
                page = meta.get("page")
                out.append(
                    Citation(
                        id=i,
                        source=str(meta.get("source", "unknown")),
                        page=page if isinstance(page, int) else None,
                        section=meta.get("section"),
                        snippet=used[i - 1].text.strip()[:240],
                    )
                )
        return out

    def _render(self, answer: str, used: Sequence[RetrievedChunk]) -> str:
        if self.gen.citation_style == "inline_source":
            return _CITE.sub(
                lambda m: f"[{short_label(used[int(m[1]) - 1])}]" if 1 <= int(m[1]) <= len(used) else m[0],
                answer,
            )
        if self.gen.citation_style == "none":
            return _CITE.sub("", answer).replace("  ", " ").strip()
        return answer

    def from_structured(self, result: LLMAnswer, used: Sequence[RetrievedChunk]) -> RAGAnswer:
        ids = [*(int(i) for i in _CITE.findall(result.answer)), *result.citations]
        return RAGAnswer(
            title=result.title.strip() or "Answer",
            answer=self._render(result.answer.strip(), used),
            citations=self._citations(ids, used),
            confidence=result.confidence,
        )

    def from_text(self, text: str, used: Sequence[RetrievedChunk]) -> RAGAnswer:
        text = text.strip()
        citations = self._citations([int(i) for i in _CITE.findall(text)], used)
        first_line = _CITE.sub("", text.split("\n", 1)[0]).strip()
        title = " ".join(first_line.split()[:8]).rstrip(".,:;") or "Answer"
        confidence = "high" if citations else ("low" if not used else "medium")
        return RAGAnswer(
            title=title, answer=self._render(text, used), citations=citations, confidence=confidence
        )
