from pathlib import Path

import pytest
import yaml

from rag.config import RAGConfig, dump_config, load_config, missing_env_vars, required_env_vars
from rag.core.exceptions import ConfigError

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# ------------------------------------------------------------------ defaults
def test_defaults_without_file():
    cfg = load_config(env={})
    assert cfg.llm.provider == "google_genai"
    assert cfg.llm.model == "gemini-1.5-flash"
    assert cfg.vector_store.type == "chroma"
    assert cfg.retrieval.strategy == "hybrid"


def test_default_yaml_matches_builtin_defaults():
    # configs/default.yaml documents every option; it must not drift from schema defaults.
    assert dump_config(load_config(CONFIGS / "default.yaml", env={})) == dump_config(RAGConfig())


@pytest.mark.parametrize("name", ["default.yaml", "local.yaml", "advanced.yaml"])
def test_sample_configs_are_valid(name):
    load_config(CONFIGS / name, env={})


def test_sample_overlays_apply():
    local = load_config(CONFIGS / "local.yaml", env={})
    assert (local.llm.provider, local.embeddings.provider, local.vector_store.type) == (
        "ollama",
        "huggingface",
        "faiss",
    )
    advanced = load_config(CONFIGS / "advanced.yaml", env={})
    assert advanced.query_transform.type == "multi_query"
    assert advanced.generation.self_check is True


# ---------------------------------------------------------------- precedence
def test_yaml_partial_merges_over_defaults(tmp_path):
    cfg = load_config(write_yaml(tmp_path, {"retrieval": {"k": 3}}), env={})
    assert cfg.retrieval.k == 3
    assert cfg.retrieval.fetch_k == 24  # untouched sibling keeps default


def test_precedence_yaml_env_cli(tmp_path):
    path = write_yaml(tmp_path, {"retrieval": {"k": 3, "fetch_k": 30}, "splitter": {"chunk_size": 500}})
    env = {"RAG__RETRIEVAL__K": "8", "RAG__SPLITTER__CHUNK_SIZE": "700"}
    cfg = load_config(path, ["retrieval.k=12"], env=env)
    assert cfg.retrieval.k == 12  # CLI beats env beats YAML
    assert cfg.splitter.chunk_size == 700  # env beats YAML
    assert cfg.retrieval.fetch_k == 30  # YAML beats default


def test_override_value_types():
    cfg = load_config(
        overrides=[
            "llm.temperature=0.7",
            "llm.structured_output=false",
            "retrieval.hybrid.weights=[0.5, 0.5]",
            "retrieval.filters={source: a.pdf}",
            "llm.max_tokens=null",
        ],
        env={},
    )
    assert cfg.llm.temperature == 0.7
    assert cfg.llm.structured_output is False
    assert cfg.retrieval.hybrid.weights == [0.5, 0.5]
    assert cfg.retrieval.filters == {"source": "a.pdf"}
    assert cfg.llm.max_tokens is None


def test_env_keys_case_insensitive_and_nested():
    cfg = load_config(env={"rag__vector_store__chroma__persist_dir": "/tmp/x", "OTHER": "1"})
    assert cfg.vector_store.chroma.persist_dir == Path("/tmp/x")


def test_config_path_from_env(tmp_path):
    path = write_yaml(tmp_path, {"app": {"name": "from-env"}})
    assert load_config(env={"RAG_CONFIG": str(path)}).app.name == "from-env"


def test_json_section_alias(tmp_path):
    cfg = load_config(write_yaml(tmp_path, {"ingestion": {"json": {"jq_schema": ".items[]"}}}), env={})
    assert cfg.ingestion.json_.jq_schema == ".items[]"
    assert dump_config(cfg)["ingestion"]["json"]["jq_schema"] == ".items[]"


def test_loader_extensions_normalized():
    cfg = load_config(overrides=["ingestion.loaders={.PDF: pymupdf}"], env={})
    assert cfg.ingestion.loaders == {"pdf": "pymupdf"}


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "overrides, fragment",
    [
        (["splitter.chunk_overlap=1000"], "chunk_overlap"),
        (["retrieval.unknown=1"], "retrieval.unknown"),
        (["embeddings.provider=anthropic"], "Anthropic does not offer"),
        (["retrieval.hybrid.weights=[0.7, 0.7]"], "sum to 1"),
        (["retrieval.hybrid.weights=[1.0]"], "exactly two"),
        (["reranker.type=llm", "reranker.top_n=10"], "top_n"),
        (["retrieval.k=30"], "fetch_k"),
        (["retrieval.search_type=threshold"], "score_threshold"),
        (["llm.provider=cohere"], "llm.provider"),
        (["llm.temperature=5"], "llm.temperature"),
        (["retrieval.parent.child_chunk_size=5000"], "child_chunk_size"),
        (["vector_store.type=qdrant", "vector_store.qdrant.path=null"], "qdrant needs"),
    ],
)
def test_invalid_configs(overrides, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_config(overrides=overrides, env={})


def test_threshold_ok_when_score_given():
    cfg = load_config(overrides=["retrieval.search_type=threshold", "retrieval.score_threshold=0.4"], env={})
    assert cfg.retrieval.score_threshold == 0.4


def test_log_level_case_insensitive():
    assert load_config(overrides=["app.log_level=debug"], env={}).app.log_level == "DEBUG"


@pytest.mark.parametrize("bad", ["retrieval.k", "=3", ""])
def test_malformed_override(bad):
    with pytest.raises(ConfigError, match="override"):
        load_config(overrides=[bad], env={})


def test_override_into_scalar_rejected():
    with pytest.raises(ConfigError, match="not a section"):
        load_config(overrides=["retrieval.k=1", "retrieval.k.x=1"], env={})
    with pytest.raises(ConfigError, match="retrieval.k"):
        load_config(overrides=["retrieval.k.x=1"], env={})


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env={})


def test_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path, env={})


def test_non_mapping_yaml(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, env={})


def test_empty_yaml_is_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path, env={}) == RAGConfig()


# --------------------------------------------------------------- credentials
def test_required_env_vars_default():
    assert set(required_env_vars(RAGConfig())) == {"GOOGLE_API_KEY"}


def test_required_env_vars_mixed_providers():
    cfg = load_config(
        overrides=[
            "llm.provider=anthropic",
            "llm.model=claude-sonnet-5-5",
            "embeddings.provider=openai",
            "vector_store.type=pgvector",
            "reranker.type=cohere",
            "observability.langsmith=true",
        ],
        env={},
    )
    assert set(required_env_vars(cfg)) == {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "PG_CONN",
        "COHERE_API_KEY",
        "LANGSMITH_API_KEY",
    }


def test_local_config_needs_no_keys():
    assert required_env_vars(load_config(CONFIGS / "local.yaml", env={})) == {}


def test_missing_env_vars():
    cfg = RAGConfig()
    assert missing_env_vars(cfg, env={}) == {"GOOGLE_API_KEY": "llm.provider=google_genai"}
    assert missing_env_vars(cfg, env={"GOOGLE_API_KEY": ""}) != {}
    assert missing_env_vars(cfg, env={"GOOGLE_API_KEY": "x"}) == {}
