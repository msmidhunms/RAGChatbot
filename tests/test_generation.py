import pytest
from langchain_core.documents import Document

from rag.config import load_config
from rag.core.exceptions import ConfigError
from rag.core.types import RetrievedChunk
from rag.generation.context import count_tokens, format_context
from rag.generation.generator import NO_ANSWER, Generator
from rag.generation.prompts import answer_prompt
from rag.generation.schemas import LLMAnswer


def chunks():
    return [
        RetrievedChunk(
            Document(
                page_content="Venus is the hottest planet.",
                metadata={"source": "docs/planets.md", "page": 2, "section": "Inner"},
            ),
            0.9,
            1,
        ),
        RetrievedChunk(Document(page_content="Mars is red.", metadata={"source": "docs/planets.md"}), 0.5, 2),
    ]


def gen(fake_llm, *overrides):
    cfg = load_config(overrides=list(overrides), env={})
    return Generator(lambda: fake_llm, cfg)


def test_format_context_numbers_and_labels():
    text, used = format_context(chunks(), 1000)
    assert text.startswith("[1] (docs/planets.md, page 2, section: Inner)\nVenus")
    assert "[2] (docs/planets.md)\nMars" in text
    assert len(used) == 2


def test_format_context_budget():
    many = chunks() * 5
    _, used = format_context(
        many, count_tokens("[1] (docs/planets.md, page 2, section: Inner)\n" + "x" * 30) + 5
    )
    assert len(used) == 1
    text, used = format_context([RetrievedChunk(Document(page_content="word " * 500), 1.0)], 10)
    assert len(used) == 1 and len(text) <= 40


def test_structured_answer_resolves_citations(fake_llm):
    fake_llm.structured["LLMAnswer"] = LLMAnswer(
        title="Hottest planet", answer="Venus is hottest [1].", citations=[1, 7], confidence="high"
    )
    answer = gen(fake_llm).generate("hottest?", chunks())
    assert answer.answer == "Venus is hottest [1]."
    assert [(c.id, c.source, c.page, c.section) for c in answer.citations] == [
        (1, "docs/planets.md", 2, "Inner")
    ]
    assert answer.confidence == "high" and answer.title == "Hottest planet"
    prompt_text = fake_llm.calls[-1][-1].content
    assert "Venus is the hottest planet." in prompt_text and "Question: hottest?" in prompt_text


def test_structured_failure_falls_back_to_text(fake_llm):
    fake_llm.responses = ["Mars is red [2]."]  # no structured response configured -> KeyError -> fallback
    answer = gen(fake_llm).generate("color of mars?", chunks())
    assert answer.answer == "Mars is red [2]."
    assert [c.id for c in answer.citations] == [2]
    assert answer.confidence == "high"


def test_text_mode_and_title(fake_llm):
    fake_llm.responses = ["Venus is the hottest planet in the solar system by far [1]\nMore detail."]
    answer = gen(fake_llm, "llm.structured_output=false").generate("q", chunks())
    assert answer.title == "Venus is the hottest planet in the solar"
    assert answer.citations[0].page == 2


def test_refuse_without_context_skips_llm(fake_llm):
    answer = gen(fake_llm).generate("q", [])
    assert answer.answer == NO_ANSWER and answer.confidence == "low"
    assert fake_llm.calls == []
    assert list(gen(fake_llm).stream("q", [])) == [NO_ANSWER]


def test_general_knowledge_without_context(fake_llm):
    fake_llm.responses = ["Not from the documents: the sky is blue."]
    answer = gen(
        fake_llm, "generation.answer_when_no_context=general_knowledge", "llm.structured_output=false"
    ).generate("sky?", [])
    assert "sky is blue" in answer.answer and answer.confidence == "low"
    assert "no relevant documents were found" in fake_llm.calls[-1][-1].content


def test_inline_source_and_none_styles(fake_llm):
    fake_llm.responses = ["Venus [1] and Mars [2] and bogus [9]."]
    inline = gen(fake_llm, "llm.structured_output=false", "generation.citation_style=inline_source")
    assert (
        inline.generate("q", chunks()).answer == "Venus [planets.md p.2] and Mars [planets.md] and bogus [9]."
    )
    none = gen(fake_llm, "llm.structured_output=false", "generation.citation_style=none")
    result = none.generate("q", chunks())
    assert result.answer == "Venus and Mars and bogus ." and result.citations == []


def test_stream_tokens(fake_llm):
    fake_llm.responses = ["Venus is hottest [1]."]
    g = gen(fake_llm)
    tokens = list(g.stream("q", chunks()))
    assert len(tokens) > 1 and "".join(tokens) == "Venus is hottest [1]."


def test_feedback_and_history_in_prompt(fake_llm):
    fake_llm.responses = ["ok [1]"]
    gen(fake_llm, "llm.structured_output=false").generate(
        "q", chunks(), history="User: hi", feedback="X is wrong"
    )
    content = fake_llm.calls[-1][-1].content
    assert content.startswith("Conversation so far:\nUser: hi")
    assert "X is wrong" in content


@pytest.mark.parametrize("name", ["default", "concise", "detailed", "strict_citations"])
def test_builtin_templates(name):
    cfg = load_config(overrides=[f"generation.prompt_template={name}"], env={})
    msgs = answer_prompt(cfg.generation).format_messages(question="q", context="c", history="", feedback="")
    assert "context" in msgs[0].content.lower()


def test_template_file(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("Answer like a pirate using {context}")
    cfg = load_config(overrides=[f"generation.prompt_template={path}"], env={})
    msgs = answer_prompt(cfg.generation).format_messages(question="q", context="C!", history="", feedback="")
    assert msgs[0].content == "Answer like a pirate using C!" and "Question: q" in msgs[1].content

    bad = tmp_path / "bad.txt"
    bad.write_text("no placeholder")
    with pytest.raises(ConfigError, match="placeholder"):
        answer_prompt(load_config(overrides=[f"generation.prompt_template={bad}"], env={}).generation)
    with pytest.raises(ConfigError, match="neither a built-in"):
        answer_prompt(load_config(overrides=["generation.prompt_template=nope"], env={}).generation)
