#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Resolve runtime operator names and compositions onto Store query scopes.

Names and compositions are different problems.  A name such as
``fused_moe_fp8`` can be normalized mechanically onto the Store's ``moe``
token.  A composite path such as GDN needs several independent query lanes so
that parent and component knowledge remain isolated instead of one replacing
the other.

The resolver therefore has three deterministic name stages (direct, adjacent
joins, corpus vote) and a small relation registry for decomposition facts that
cannot be recovered from spelling alone.  Ambiguous votes stay unresolved.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import query_wiki as query


MAX_JOIN_LENGTH = 4
MIN_TOKEN_LENGTH = 3
VOTE_MARGIN = 1.5
PHRASE_WEIGHT = 2.5
MAX_CORPUS_DOCUMENTS = 10_000
MAX_CORPUS_TEXT_BYTES = 64 * 1024 * 1024

_NOISE = frozenset({
    "fused", "fusion", "kernel", "kernels", "op", "ops", "operator",
    "fwd", "bwd", "forward", "backward", "fp8", "fp16", "fp32",
    "bf16", "fp4", "nvfp4", "mxfp4", "int8", "int4", "e4m3",
    "e5m2", "v1", "v2", "v3", "dynamic", "static", "batched",
    "quantized",
})

Axis = Literal["operator", "family"]
Role = Literal["primary", "component", "related"]
Confidence = Literal["declared", "direct", "voted"]


@dataclass(frozen=True)
class OperatorScope:
    """One independent Store query lane."""

    role: Role
    axis: Axis
    value: str
    confidence: Confidence
    source_term: str


@dataclass(frozen=True)
class RelationGroup:
    """A directed parent-to-component relation spelling cannot prove."""

    parent_aliases: tuple[str, ...]
    scopes: tuple[tuple[Axis, str], ...]


# A composite GDN path can expose recurrent/state-space and convolution work as
# separate optimization units. These are query relations, not a claim that
# every component is nested inside one kernel implementation.
RELATION_GROUPS = (
    RelationGroup(
        parent_aliases=(
            "gdn", "gated delta net", "gated delta rule",
        ),
        scopes=(
            ("operator", "gdn"),
            ("family", "gdn"),
            ("family", "mamba"),
            ("family", "conv"),
        ),
    ),
)

# Component/API spellings resolve only to their own Store lane. They are kept
# separate from RELATION_GROUPS so a standalone component cannot activate its
# parent or sibling operators.
DECLARED_SCOPE_ALIASES: dict[str, tuple[Axis, str]] = {
    "causal conv1d": ("family", "conv"),
    "causal_conv1d": ("family", "conv"),
    "causal conv1d update": ("family", "conv"),
    "causal_conv1d_update": ("family", "conv"),
}


def fold(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def _tokens(name: str) -> tuple[str, ...]:
    # Preserve the Runtime's spelling while also handling common CamelCase API
    # names.  Acronym boundaries intentionally remain one token.
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return tuple(
        part for part in re.split(r"[^A-Za-z0-9]+", separated.lower()) if part
    )


def _candidates(tokens: Iterable[str]) -> list[str]:
    parts = list(tokens)
    out: list[str] = []
    for size in range(min(MAX_JOIN_LENGTH, len(parts)), 0, -1):
        for start in range(len(parts) - size + 1):
            out.append("-".join(parts[start:start + size]))
    return out


def _is_noise_sequence(value: str) -> bool:
    """Return whether a compact affix is composed only of known decorations."""
    if not value:
        return True
    decorations = sorted(
        {fold(token) for token in _NOISE if token}, key=len, reverse=True
    )
    reachable = {0}
    for start in range(len(value) + 1):
        if start not in reachable:
            continue
        for decoration in decorations:
            if value.startswith(decoration, start):
                reachable.add(start + len(decoration))
    return len(value) in reachable


class OperatorScopeResolver:
    """Map caller terms to the Store without collapsing composite operators."""

    def __init__(
        self,
        *,
        operators: Iterable[str],
        families: Iterable[str],
        document_loader: Callable[[], Iterable[tuple[str, str]]],
    ) -> None:
        self._operators = {fold(value): value for value in operators if value}
        self._families = {fold(value): value for value in families if value}
        self._document_loader = document_loader
        self._documents: list[tuple[str, str, frozenset[str]]] | None = None
        self._document_text_bytes = 0
        self._family_size: Counter[str] = Counter()
        self._document_frequency: Counter[str] = Counter()
        self._relations = {
            fold(alias): group
            for group in RELATION_GROUPS
            for alias in group.parent_aliases
        }
        self._scope_aliases = {
            fold(alias): scope for alias, scope in DECLARED_SCOPE_ALIASES.items()
        }
        active_relation_groups = (
            group for group in RELATION_GROUPS
            if any(fold(alias) in self._operators or fold(alias) in self._families
                   for alias in group.parent_aliases)
            or any(self._known(axis, value) for axis, value in group.scopes)
        )
        self._invalid_relation_scopes = tuple(sorted({
            (axis, value)
            for group in active_relation_groups
            for axis, value in group.scopes
            if not self._known(axis, value)
        }))
        if self._invalid_relation_scopes:
            warnings.warn(
                "operator relation registry references unknown Store scopes: %s"
                % ", ".join("%s:%s" % row
                            for row in self._invalid_relation_scopes),
                RuntimeWarning,
                stacklevel=2,
            )

    @classmethod
    def from_store(cls, root: Path) -> "OperatorScopeResolver":
        vocabulary = query.vocab(query.load_index(root / "kernel_wiki")["records"])

        def load_documents() -> Iterable[tuple[str, str]]:
            index = query.load_index(root / "kernel_wiki")
            for record in index.get("records") or []:
                scope = ((record.get("retrieval") or {}).get("scope") or {})
                family = scope.get("operator_family")
                if not family:
                    continue
                # Public indexes already carry a deterministic ``search_text``
                # projection. Reuse it directly and never load the served JSON
                # payload shards merely to classify an operator spelling.
                text = " ".join(filter(None, (
                    str(record.get("id") or ""),
                    str(record.get("title") or ""),
                    str(record.get("search_text") or ""),
                )))
                yield str(family), text

        return cls(
            operators=vocabulary.get("operator") or (),
            families=vocabulary.get("family") or (),
            document_loader=load_documents,
        )

    def resolve_many(
        self,
        primary_terms: Iterable[str],
        component_terms: Iterable[str],
        *,
        limit: int = 8,
    ) -> tuple[list[OperatorScope], list[str]]:
        scopes: list[OperatorScope] = []
        unresolved: list[str] = []
        seen: set[tuple[str, str]] = set()

        def add(scope: OperatorScope) -> None:
            key = (scope.axis, scope.value)
            if key not in seen and len(scopes) < limit:
                seen.add(key)
                scopes.append(scope)

        for role, terms in (("primary", primary_terms),
                            ("component", component_terms)):
            for raw in terms:
                term = str(raw).strip()
                if not term:
                    continue
                resolved = self.resolve(term, role=role)
                if resolved is not None:
                    add(resolved)
                    # Some Store tokens intentionally exist on both axes.  The
                    # current hardware/DSL cell may populate only one of them,
                    # so keep both as isolated lanes instead of guessing.
                    key = fold(resolved.value)
                    if resolved.axis == "operator" and key in self._families:
                        add(OperatorScope(
                            role="related", axis="family",
                            value=self._families[key], confidence="direct",
                            source_term=term,
                        ))
                    elif resolved.axis == "family" and key in self._operators:
                        add(OperatorScope(
                            role="related", axis="operator",
                            value=self._operators[key], confidence="direct",
                            source_term=term,
                        ))
                relation = self._relations.get(fold(term))
                if relation is not None:
                    for axis, value in relation.scopes:
                        if self._known(axis, value):
                            add(OperatorScope(
                                role="related", axis=axis, value=value,
                                confidence="declared", source_term=term,
                            ))
                elif resolved is None:
                    unresolved.append(term)
        return scopes, unresolved

    def resolve(self, term: str, *, role: Role = "primary") -> OperatorScope | None:
        # Compare the complete compact spelling first. This handles long API
        # names whose only differences are case, underscores, or hyphens
        # without depending on the bounded adjacent-token search below.
        whole = fold(term)
        if whole in self._operators:
            return OperatorScope(role, "operator", self._operators[whole],
                                 "direct", term)
        if whole in self._families:
            return OperatorScope(role, "family", self._families[whole],
                                 "direct", term)
        declared = self._scope_aliases.get(whole)
        if declared is not None and self._known(*declared):
            return OperatorScope(role, declared[0], declared[1],
                                 "declared", term)
        tokens = _tokens(term)
        if not tokens:
            return None
        for candidate in _candidates(tokens):
            key = fold(candidate)
            if key in self._operators:
                return OperatorScope(role, "operator", self._operators[key],
                                     "direct", term)
            if key in self._families:
                return OperatorScope(role, "family", self._families[key],
                                     "direct", term)
        embedded = self._resolve_decorated(whole, role, term)
        if embedded is not None:
            return embedded
        family = self._vote(tokens)
        if family is None:
            return None
        return OperatorScope(role, "family", family, "voted", term)

    def _resolve_decorated(
        self, whole: str, role: Role, source_term: str,
    ) -> OperatorScope | None:
        """Resolve a vocabulary token wrapped only in harmless API decorations.

        CamelCase acronym runs such as ``FusedMoEFP8`` and
        ``QuantizedGEMMKernel`` cannot be split reliably without knowing the
        vocabulary. Match the vocabulary first, then require every character
        outside it to be a reviewed decoration. Ambiguous longest matches fail
        closed instead of guessing.
        """
        matches: list[tuple[int, str, Axis, str]] = []
        for axis, table in (("operator", self._operators),
                            ("family", self._families)):
            for key, value in table.items():
                if len(key) < MIN_TOKEN_LENGTH:
                    continue
                start = whole.find(key)
                while start >= 0:
                    end = start + len(key)
                    if (_is_noise_sequence(whole[:start])
                            and _is_noise_sequence(whole[end:])):
                        matches.append((len(key), key, axis, value))
                    start = whole.find(key, start + 1)
        if not matches:
            return None
        longest = max(length for length, _key, _axis, _value in matches)
        matches = [row for row in matches if row[0] == longest]
        keys = {key for _length, key, _axis, _value in matches}
        if len(keys) != 1:
            return None
        # The same token can be indexed as both an operator and a family. Keep
        # operator as the primary address; resolve_many adds the family as an
        # isolated related lane.
        selected = next(
            (row for row in matches if row[2] == "operator"), matches[0]
        )
        return OperatorScope(role, selected[2], selected[3], "direct", source_term)

    def _known(self, axis: Axis, value: str) -> bool:
        table = self._operators if axis == "operator" else self._families
        return fold(value) in table

    def _ensure_documents(self) -> None:
        if self._documents is not None:
            return
        self._documents = []
        for family, text in self._document_loader():
            text_bytes = len(text.encode("utf-8"))
            if (len(self._documents) >= MAX_CORPUS_DOCUMENTS
                    or self._document_text_bytes + text_bytes
                    > MAX_CORPUS_TEXT_BYTES):
                raise ValueError(
                    "operator corpus exceeds the in-process vote budget "
                    "(%d documents or %d bytes); use a structured operator "
                    "scope or raise the reviewed limit"
                    % (MAX_CORPUS_DOCUMENTS, MAX_CORPUS_TEXT_BYTES)
                )
            words = frozenset(re.findall(r"[a-z0-9]+", text))
            self._documents.append((family, text, words))
            self._document_text_bytes += text_bytes
            self._family_size[family] += 1
            self._document_frequency.update(words)

    def _vote(self, tokens: tuple[str, ...]) -> str | None:
        terms = tuple(
            token for token in tokens
            if len(token) >= MIN_TOKEN_LENGTH and token not in _NOISE
        )
        if not terms:
            return None
        self._ensure_documents()
        assert self._documents is not None
        if not self._documents:
            return None
        phrases = [
            candidate for candidate in _candidates(tokens)
            if "-" in candidate
            and all(part not in _NOISE for part in candidate.split("-"))
        ]
        scores: dict[str, float] = {}
        total_documents = len(self._documents)
        for family, text, words in self._documents:
            score = 0.0
            for term in terms:
                if term in words:
                    score += math.log1p(
                        total_documents / (1 + self._document_frequency[term])
                    )
            for phrase in phrases:
                if phrase.replace("-", " ") in text:
                    score += PHRASE_WEIGHT * min(
                        math.log1p(
                            total_documents
                            / (1 + self._document_frequency[part])
                        )
                        for part in phrase.split("-")
                    )
            if score:
                scores[family] = scores.get(family, 0.0) + score
        if not scores:
            return None
        ranked = sorted(
            (
                (score / math.sqrt(self._family_size[family]), family)
                for family, score in scores.items()
            ),
            reverse=True,
        )
        best, family = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if runner_up and best < VOTE_MARGIN * runner_up:
            return None
        return family


__all__ = ["OperatorScope", "OperatorScopeResolver", "fold"]
