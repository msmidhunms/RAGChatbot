import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag_chatbot.config.schema import MemoryConfig
from rag_chatbot.generation.generator import NO_ANSWER
from rag_chatbot.generation.schemas import GradeResult, GroundednessResult, LLMAnswer
from rag_chatbot.memory.history import build_checkpointer, format_history, select_history, update_summary
from rag_chatbot.pipeline import RAGPipeline


@pytest.fixture
def make_pipeline(make_cfg, keyword_embeddings, fake_llm, corpus):
    def build(*overrides, ingest=True):
        cfg = make_cfg(
            "ingestion.cleaning.min_chars=10",
            "splitter.chunk_size=300",
            "splitter.chunk_overlap=30",
            "memory.checkpointer=sqlite",
            *overrides,
        )
        pipe = RAGPipeline(cfg, llm=fake_llm, embeddings=keyword_embeddings)
        if ingest:
            pipe.ingest([str(corpus)])
        return pipe

    return build


def answer(text="Venus is the hottest planet [1].", cites=(1,)):
    return LLMAnswer(title="Hot", answer=text, citations=list(cites), confidence="high")


# ----------------------------------------------------------------- query path
def test_query_basic(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer()
    result = make_pipeline().query("Which planet is the hottest?")
    assert result.answer.answer == "Venus is the hottest planet [1]."
    assert result.answer.citations[0].source.endswith("sample.md")
    assert result.chunks and "Venus" in result.chunks[0].text
    assert set(result.timings) == {"condense", "transform", "retrieve", "generate", "finalize"}
    assert result.standalone_question == "Which planet is the hottest?"
    assert result.queries == ["Which planet is the hottest?"]
    assert result.to_dict()["sources"][0]["rank"] == 1


def test_query_refuses_when_nothing_found(make_pipeline, fake_llm):
    result = make_pipeline(ingest=False).query("anything?")
    assert result.answer.answer == NO_ANSWER and fake_llm.calls == []


def test_query_with_filter(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer("Shipping takes 3-5 days [1].")
    result = make_pipeline().query("how long", flt={"file_type": "json"})
    assert result.chunks and all(c.metadata["file_type"] == "json" for c in result.chunks)


def test_usage_counted(make_pipeline, fake_llm):
    fake_llm.responses = ["Venus [1]"]
    result = make_pipeline("llm.structured_output=false").query("hottest planet")
    assert result.usage["llm_calls"] == 1 and result.usage["total_tokens"] == 15


# ---------------------------------------------------------------- optional nodes
def test_postprocess_node_added(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer()
    result = make_pipeline("compression.type=redundant_filter").query("hottest planet")
    assert "postprocess" in result.timings


def test_grade_keeps_relevant_chunks(make_pipeline, fake_llm):
    fake_llm.structured["GradeResult"] = GradeResult(relevant=[1])
    fake_llm.structured["LLMAnswer"] = answer()
    result = make_pipeline("generation.grade_documents=true").query("hottest planet venus")
    assert len(result.chunks) == 1 and "grade" in result.timings


def test_grade_failure_rewrites_query_once(make_pipeline, fake_llm):
    fake_llm.structured["GradeResult"] = [GradeResult(relevant=[]), GradeResult(relevant=[1])]
    fake_llm.structured["LLMAnswer"] = answer()
    fake_llm.responses = ["Venus surface temperature"]  # the rewrite
    result = make_pipeline("generation.grade_documents=true").query("hot one?")
    assert result.queries == ["Venus surface temperature"]
    assert len(result.chunks) == 1
    assert result.answer.answer.startswith("Venus")


def test_grade_gives_up_after_max_retries(make_pipeline, fake_llm):
    fake_llm.structured["GradeResult"] = GradeResult(relevant=[])
    fake_llm.responses = ["rewritten"]
    result = make_pipeline("generation.grade_documents=true", "generation.max_retries=1").query("??")
    assert result.answer.answer == NO_ANSWER and result.chunks == []


def test_self_check_regenerates_with_feedback(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = [answer("Venus is 900 degrees [1]."), answer()]
    fake_llm.structured["GroundednessResult"] = [
        GroundednessResult(grounded=False, issues="900 degrees is not in the context"),
        GroundednessResult(grounded=True),
    ]
    result = make_pipeline("generation.self_check=true").query("hottest planet")
    assert result.answer.answer == "Venus is the hottest planet [1]."
    assert result.answer.grounded is True
    regen_prompt = [m for m in fake_llm.calls if any("900 degrees is not" in x.content for x in m)]
    assert regen_prompt, "feedback should be passed to the second generation"


def test_self_check_gives_up(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer("made up [1]")
    fake_llm.structured["GroundednessResult"] = GroundednessResult(grounded=False, issues="bad")
    result = make_pipeline("generation.self_check=true", "generation.max_retries=1").query("hottest planet")
    assert result.answer.grounded is False and result.answer.answer == "made up [1]"


# ----------------------------------------------------------------------- memory
def test_chat_memory_condenses_follow_up(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer()
    pipe = make_pipeline()
    sid = pipe.new_session_id()
    first = pipe.chat("Which planet is the hottest?", sid)
    assert first.standalone_question == "Which planet is the hottest?"

    fake_llm.responses = ["What is the atmosphere of Venus made of?"]
    second = pipe.chat("What is its atmosphere made of?", sid)
    assert second.standalone_question == "What is the atmosphere of Venus made of?"
    condense_prompt = fake_llm.calls[-2][-1].content
    assert "User: Which planet is the hottest?" in condense_prompt
    assert "Assistant: Venus is the hottest planet [1]." in condense_prompt

    history = pipe.history(sid)
    assert [type(m) for m in history] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    pipe.clear_session(sid)
    assert pipe.history(sid) == []


def test_chat_sessions_persist_in_sqlite(make_pipeline, fake_llm, keyword_embeddings):
    fake_llm.structured["LLMAnswer"] = answer()
    pipe = make_pipeline()
    pipe.chat("hottest planet?", "s1")
    reopened = RAGPipeline(pipe.cfg, llm=fake_llm, embeddings=keyword_embeddings)
    assert len(reopened.history("s1")) == 2
    assert reopened.history("other") == []


def test_memory_none_skips_condense(make_pipeline, fake_llm):
    fake_llm.structured["LLMAnswer"] = answer()
    pipe = make_pipeline("memory.type=none")
    pipe.chat("hottest planet?", "s")
    calls = len(fake_llm.calls)
    pipe.chat("and the coldest?", "s")
    assert len(fake_llm.calls) == calls + 1  # only generation, no condense call


def test_history_helpers():
    msgs = [HumanMessage(content=f"q{i}") if i % 2 == 0 else AIMessage(content=f"a{i}") for i in range(10)]
    window = MemoryConfig(type="window", window_size=2)
    assert [m.content for m in select_history(msgs, window)] == ["q6", "a7", "q8", "a9"]
    assert select_history(msgs, MemoryConfig(type="none")) == []
    assert len(select_history(msgs, MemoryConfig(type="buffer"))) == 10
    summary_cfg = MemoryConfig(type="summary", window_size=2)
    assert format_history(msgs[:2], summary_cfg, "earlier stuff").startswith(
        "Summary of earlier conversation"
    )


def test_update_summary(fake_llm):
    fake_llm.responses = ["They discussed q0..a5."]
    msgs = [HumanMessage(content=f"q{i}") if i % 2 == 0 else AIMessage(content=f"a{i}") for i in range(10)]
    cfg = MemoryConfig(type="summary", window_size=2)
    summary, upto = update_summary(msgs, cfg, "", 0, lambda: fake_llm)
    assert summary == "They discussed q0..a5." and upto == 6
    assert "User: q0" in fake_llm.calls[-1][-1].content and "q6" not in fake_llm.calls[-1][-1].content
    assert update_summary(msgs, cfg, summary, 6, lambda: fake_llm) == (summary, 6)
    assert update_summary(msgs, MemoryConfig(type="window"), "", 0, lambda: fake_llm) == ("", 0)


def test_memory_checkpointer(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    assert isinstance(build_checkpointer(MemoryConfig(checkpointer="memory")), InMemorySaver)
    saver = build_checkpointer(MemoryConfig(checkpointer="sqlite", path=tmp_path / "x" / "s.sqlite"))
    assert type(saver).__name__ == "SqliteSaver" and (tmp_path / "x" / "s.sqlite").exists()


# -------------------------------------------------------------------- streaming
def test_stream_chat_tokens(make_pipeline, fake_llm):
    fake_llm.responses = ["Venus is the hottest planet [1]."]
    fake_llm.structured["GradeResult"] = GradeResult(relevant=[1, 2])
    pipe = make_pipeline("generation.grade_documents=true")
    out = list(pipe.stream_chat("hottest planet?", "s1"))
    tokens, result = out[:-1], out[-1]
    assert len(tokens) > 1 and "".join(tokens) == "Venus is the hottest planet [1]."
    assert result.answer.answer == "Venus is the hottest planet [1]."
    assert result.answer.citations[0].id == 1
    assert len(pipe.history("s1")) == 2


def test_stream_refusal_yields_text(make_pipeline):
    out = list(make_pipeline(ingest=False).stream_chat("x?"))
    assert out[0] == NO_ANSWER and out[-1].answer.answer == NO_ANSWER


# ------------------------------------------------------------------------ stats
def test_stats_and_store_ops(make_pipeline, corpus):
    pipe = make_pipeline()
    stats = pipe.stats()
    assert stats["sources"] == 5 and stats["chunks"] == stats["vectors"] > 0
    md = next(s for s in stats["per_source"] if s.endswith("sample.md"))
    assert pipe.delete_source(md) > 0
    assert pipe.stats()["sources"] == 4
    pipe.reset_store()
    assert pipe.stats()["chunks"] == 0
