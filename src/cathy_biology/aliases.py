from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import httpx

from cathy_biology.models import ResolvedEntity
from cathy_biology.utils import ensure_directory


class GeneAliasResolver:
    QUERY_URL = "https://mygene.info/v3/query"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = ensure_directory(cache_dir)
        self.cache_path = self.cache_dir / "alias_cache.json"
        self._cache = self._load_cache()

    def resolve_symbols(
        self,
        symbols: Iterable[str],
        preferred_symbols: set[str] | None = None,
        extra_aliases: dict[str, list[str]] | None = None,
    ) -> dict[str, ResolvedEntity]:
        resolved: dict[str, ResolvedEntity] = {}
        for symbol in symbols:
            normalized = self.resolve_symbol(symbol, preferred_symbols=preferred_symbols, extra_aliases=extra_aliases)
            resolved[symbol] = normalized
        return resolved

    def resolve_symbol(
        self,
        symbol: str,
        preferred_symbols: set[str] | None = None,
        extra_aliases: dict[str, list[str]] | None = None,
    ) -> ResolvedEntity:
        query = symbol.strip()
        if not query:
            return ResolvedEntity(canonical_symbol=symbol, aliases=[], entity_type="unknown", sources=["empty-query"])
        if preferred_symbols and query.upper() in preferred_symbols:
            return ResolvedEntity(canonical_symbol=query.upper(), aliases=[query], sources=["preferred-symbol"])
        cached = self._cache.get(query)
        if cached is not None:
            return ResolvedEntity.model_validate(cached)

        local_match = self._resolve_from_alias_hints(query, preferred_symbols, extra_aliases or {})
        if local_match is not None:
            self._cache[query] = local_match.model_dump()
            self._write_cache()
            return local_match

        try:
            response = httpx.get(
                self.QUERY_URL,
                params={
                    "q": query,
                    "species": "human",
                    "size": 10,
                    "fields": "symbol,alias,name,retired,uniprot",
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            hits = payload.get("hits", [])
        except Exception:
            hits = []

        resolved = self._pick_best_hit(query, hits, preferred_symbols)
        self._cache[query] = resolved.model_dump()
        self._write_cache()
        return resolved

    def _resolve_from_alias_hints(
        self,
        query: str,
        preferred_symbols: set[str] | None,
        extra_aliases: dict[str, list[str]],
    ) -> ResolvedEntity | None:
        query_upper = query.upper()
        for canonical, aliases in extra_aliases.items():
            if query_upper == canonical.upper() or query_upper in {alias.upper() for alias in aliases}:
                if preferred_symbols and canonical.upper() not in preferred_symbols:
                    continue
                return ResolvedEntity(
                    canonical_symbol=canonical.upper(),
                    aliases=sorted({query, *aliases}),
                    sources=["prompt-alias-hint"],
                )
        return None

    def _pick_best_hit(
        self,
        query: str,
        hits: list[dict],
        preferred_symbols: set[str] | None,
    ) -> ResolvedEntity:
        query_upper = query.upper()
        best_score = -1
        best_hit: dict | None = None
        for hit in hits:
            symbol = str(hit.get("symbol", "")).upper()
            aliases = {str(alias).upper() for alias in _as_list(hit.get("alias"))}
            retired = {str(alias).upper() for alias in _as_list(hit.get("retired"))}
            score = 0
            if preferred_symbols and symbol in preferred_symbols:
                score += 20
            if symbol == query_upper:
                score += 100
            if query_upper in aliases:
                score += 80
            if query_upper in retired:
                score += 60
            if symbol.startswith(query_upper) or query_upper.startswith(symbol):
                score += 10
            if score > best_score:
                best_score = score
                best_hit = hit

        if best_hit is None:
            return ResolvedEntity(canonical_symbol=query_upper, aliases=[query], sources=["unresolved"])

        aliases = sorted({query, *map(str, _as_list(best_hit.get("alias"))), str(best_hit.get("symbol", query_upper))})
        return ResolvedEntity(
            canonical_symbol=str(best_hit.get("symbol", query_upper)).upper(),
            aliases=aliases,
            sources=["mygene"],
        )

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _write_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
