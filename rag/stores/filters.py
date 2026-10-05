"""Store-agnostic metadata filters.

A filter is a dict of ``field: value`` (equality) or ``field: {op: value}``
with ``op`` in ``$eq $ne $in $nin $gt $gte $lt $lte``. All conditions must
hold. Each backend translates this into its own query syntax; ``matches``
evaluates it in Python for backends without native filtering.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rag.core.exceptions import ConfigError

OPERATORS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte")


def normalize_filter(flt: Mapping[str, Any] | None) -> list[tuple[str, str, Any]]:
    """Flatten into ``(field, op, value)`` conditions, validating operators."""
    conditions: list[tuple[str, str, Any]] = []
    for field, cond in (flt or {}).items():
        if isinstance(cond, Mapping):
            for op, value in cond.items():
                if op not in OPERATORS:
                    raise ConfigError(f"unknown filter operator {op!r} on {field!r}; use one of {OPERATORS}")
                if op in ("$in", "$nin") and not isinstance(value, (list, tuple)):
                    raise ConfigError(f"filter {field!r} {op} needs a list")
                conditions.append((field, op, value))
        else:
            conditions.append((field, "$eq", cond))
    return conditions


def _check(actual: Any, op: str, value: Any) -> bool:
    if op == "$eq":
        return actual == value
    if op == "$ne":
        return actual != value
    if op == "$in":
        return actual in value
    if op == "$nin":
        return actual not in value
    if actual is None:
        return False
    try:
        if op == "$gt":
            return actual > value
        if op == "$gte":
            return actual >= value
        if op == "$lt":
            return actual < value
        return actual <= value
    except TypeError:
        return False


def matches(metadata: Mapping[str, Any], flt: Mapping[str, Any] | None) -> bool:
    return all(_check(metadata.get(f), op, v) for f, op, v in normalize_filter(flt))


def to_chroma(flt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    clauses = [{f: {op: v}} for f, op, v in normalize_filter(flt)]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def to_pgvector(flt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    # langchain_postgres accepts the same operator dialect as Chroma
    return to_chroma(flt)


def to_qdrant(flt: Mapping[str, Any] | None, payload_key: str = "metadata") -> Any:
    conditions = normalize_filter(flt)
    if not conditions:
        return None
    from qdrant_client import models

    must: list[Any] = []
    must_not: list[Any] = []
    ranges: dict[str, dict[str, Any]] = {}
    for field, op, value in conditions:
        key = f"{payload_key}.{field}"
        if op == "$eq":
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        elif op == "$ne":
            must_not.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        elif op == "$in":
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        elif op == "$nin":
            must_not.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            ranges.setdefault(key, {})[op[1:]] = value
    for key, bounds in ranges.items():
        must.append(models.FieldCondition(key=key, range=models.Range(**bounds)))
    return models.Filter(must=must or None, must_not=must_not or None)
