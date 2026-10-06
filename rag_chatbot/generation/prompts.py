"""Prompt templates.

Answer prompts are selected by ``generation.prompt_template``: a registered
name (default, concise, detailed, strict_citations) or a path to a text file
whose content becomes the system message (it must contain ``{context}``).
Helper prompts used by retrieval and the graph live here too, so every
LLM instruction in the system can be read and tuned in one place.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from rag_chatbot.config.schema import GenerationConfig
from rag_chatbot.core.exceptions import ConfigError
from rag_chatbot.core.registry import Registry

BASE_SYSTEM = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "The context is a numbered list of excerpts from the user's documents."
)

STYLE: Registry[str] = Registry("prompt template")
STYLE.add("default", "Answer clearly and directly in a few sentences or a short list.")
STYLE.add("concise", "Answer in at most two sentences. No preamble.")
STYLE.add(
    "detailed",
    "Give a thorough, well-structured answer with short paragraphs or bullet points, "
    "covering every relevant detail in the context.",
)
STYLE.add(
    "strict_citations",
    "Every sentence must be supported by the context and end with its citation. "
    "Do not add any information that is not explicitly stated in the context.",
)

CITATION_RULES = {
    "numeric": "Cite supporting excerpts with their numbers in square brackets, e.g. [1] or [2][3].",
    "inline_source": "Cite supporting excerpts with their numbers in square brackets, e.g. [1]; "
    "they will be shown to the user as source names.",
    "none": "Do not include citations.",
}

NO_CONTEXT_RULES = {
    "refuse": "If the context does not contain the answer, say you could not find it in the documents. "
    "Never invent facts.",
    "general_knowledge": "If the context does not contain the answer, you may answer from general knowledge, "
    "but say clearly that the answer is not based on the documents.",
}

HUMAN = "{history}Context:\n{context}\n\nQuestion: {question}{feedback}"


def answer_prompt(cfg: GenerationConfig) -> ChatPromptTemplate:
    name = cfg.prompt_template
    if name in STYLE:
        system = " ".join(
            [
                BASE_SYSTEM,
                STYLE.get(name),
                CITATION_RULES[cfg.citation_style],
                NO_CONTEXT_RULES[cfg.answer_when_no_context],
            ]
        )
        return ChatPromptTemplate.from_messages([("system", system), ("human", HUMAN)])
    path = Path(name)
    if not path.is_file():
        raise ConfigError(
            f"generation.prompt_template '{name}' is neither a built-in template "
            f"({', '.join(STYLE.available())}) nor an existing file"
        )
    text = path.read_text(encoding="utf-8")
    if "{context}" not in text:
        raise ConfigError(f"prompt file {path} must contain the {{context}} placeholder")
    human = HUMAN if "{question}" not in text else "{history}{question}{feedback}"
    return ChatPromptTemplate.from_messages([("system", text), ("human", human)])


def _p(system: str, human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system), ("human", human)])


CONDENSE = _p(
    "Rewrite the user's follow-up question as a standalone question that can be understood "
    "without the conversation. Keep names and specifics. Return only the question.",
    "Conversation:\n{history}\n\nFollow-up question: {question}",
)

REWRITE = _p(
    "Rewrite the question to make it a better search query for a document search engine: "
    "expand abbreviations, add likely keywords, remove filler. Return only the query.",
    "{history}Question: {question}",
)

MULTI_QUERY = _p(
    "Generate {n} different search queries that together cover the user's question from different "
    "angles (synonyms, sub-questions, related terms). Each query must stand alone.",
    "{history}Question: {question}",
)

HYDE = _p(
    "Write a short passage (3-5 sentences) that would plausibly answer the question, "
    "as if taken from a reference document. It is used only to search for real documents.",
    "{history}Question: {question}",
)

STEP_BACK = _p(
    "Write a more general 'step-back' question about the underlying concept or principle behind "
    "the user's question. Return only that question.",
    "{history}Question: {question}",
)

GRADE = _p(
    "You judge whether document excerpts are relevant to a question. An excerpt is relevant if it "
    "contains information that helps answer the question, even partially.",
    "Question: {question}\n\nExcerpts:\n{context}\n\nReturn the numbers of the relevant excerpts.",
)

SELF_CHECK = _p(
    "You verify answers. Decide whether every claim in the answer is supported by the context. "
    "List unsupported claims in 'issues'.",
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}",
)

EXTRACT = _p(
    "Extract the sentences from the excerpt that are relevant to the question, verbatim. "
    "If nothing is relevant, return exactly NO_OUTPUT.",
    "Question: {question}\n\nExcerpt:\n{text}",
)

RERANK = _p(
    "Score how relevant each excerpt is to the question from 0 (irrelevant) to 10 (directly answers it).",
    "Question: {question}\n\nExcerpts:\n{context}",
)

SUMMARY = _p(
    "Update the running summary of a conversation with the new messages. Keep facts, names, "
    "decisions and open questions. At most 8 sentences.",
    "Current summary:\n{summary}\n\nNew messages:\n{history}",
)

JUDGE = _p(
    "You are a strict evaluator of a question-answering system. Score each criterion from 0.0 to 1.0.\n"
    "- faithfulness: the answer's claims are supported by the context\n"
    "- answer_relevance: the answer addresses the question\n"
    "- correctness: the answer agrees with the reference answer (1.0 if no reference is given)\n"
    "- context_recall: share of the facts in the reference answer that appear in the context "
    "(1.0 if no reference is given)",
    "Question: {question}\n\nContext:\n{context}\n\nReference answer: {reference}\n\nAnswer:\n{answer}",
)

GENERATE_QA = _p(
    "You write evaluation questions for a document question-answering system. Using only the excerpt, "
    "write one specific question that the excerpt answers, the answer, and 1-2 short quotes copied "
    "verbatim from the excerpt that prove the answer. Do not ask about formatting or the document itself.",
    "Excerpt:\n{context}",
)

GENERATE_MULTIHOP = _p(
    "You write evaluation questions that need two documents. Write one question that can only be "
    "answered by combining a fact from excerpt A with a fact from excerpt B, the answer, and two "
    "short verbatim quotes: the first copied from A, the second from B.",
    "Excerpt A:\n{context_a}\n\nExcerpt B:\n{context_b}",
)

GENERATE_UNANSWERABLE = _p(
    "You write evaluation questions that a document collection cannot answer. The excerpts show what "
    "the collection is about. Write one plausible, specific question on the same subject whose answer "
    "is NOT contained in any excerpt (for example a number, name or policy that is never mentioned).",
    "Excerpts:\n{context}",
)

GENERATE_FOLLOWUP = _p(
    "Turn the question into a two-turn conversation: an opening question that introduces the subject, "
    "then a follow-up that refers back to it with a pronoun (it, its, they, that) instead of naming it.",
    "Question: {question}",
)
