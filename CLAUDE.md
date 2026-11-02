# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`ehds_ontomap` maps a list of free-text terms onto a `code | term` ontology
(TSV or Parquet) with one of three methods: `string-match`, `levenshtein`,
`biencoder`. Part of the DataTools4Heart (DT4H) EU project, so code, docs and
repo hygiene must be release quality.

## Working agreement

- **Dependencies**: do not run `uv add`, `uv remove`, `uv sync` or
  `pip install`. Propose the exact `uv add` command, with version bounds and a
  reason for each package, and the maintainer runs it.
- **Git**: do not push, open PRs or change GitHub settings. Say when a change
  is ready to commit or push; the maintainer does it.
- **Porting**: method implementations come from
  `../dt4h-annotation-api/app/src/nel/` (MIT). Never modify that repository.
  Keep the attribution header in every adapted file.

## Commands

```bash
uv run pytest -m "not model"   # offline suite
uv run pytest -m model         # tests that download a Hugging Face model
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Layout

```
src/ehds_ontomap/
  ontology.py   Ontology: load/validate code|term from TSV or Parquet
  terms.py      load_terms(): every accepted input shape -> TermList(ids, terms)
  normalize.py  normalize_text / Normalizer (lexical methods only)
  result.py     Candidate, QueryResult, MappingResult, select_candidates()
  mapper.py     OntologyMapper / map_terms(): the public entry point
  methods/
    base.py         Matcher ABC, TermIndex, rank_rows() + _shortlist()
    string_match.py StringMatch
    levenshtein.py  Levenshtein (rapidfuzz cdist)
    biencoder.py    BiEncoder, TransformerEncoder, DEFAULT_MODELS (lazy torch import)
  errors.py     OntomapError, OntologyError, InputError
```

## Design rules

- Every method yields raw `(code, term, score)` triples per query and then
  goes through `result.select_candidates`. Threshold, `top_k`, deduplication
  by code, ties and ranking must never be reimplemented inside a method.
- No input is dropped: unmatched queries keep a row with null candidate
  fields. Never use a sentinel code such as `NO_MAP`.
- Homonyms (one term, several codes) are all returned; never keep only the
  first row.
- Ontology TSVs are read as strings with quoting disabled, so leading zeros,
  quotes and the literal string `"NA"` survive.
- Normalisation applies to lexical methods only, never to bi-encoder input.
- Default bi-encoders mirror the `nel:` section of
  `../dt4h-annotation-api/app/model_manager/default_registry.yaml`
  (ClinLinker-KB-P for `es`, SapBERT XLM-R large for the other languages).
  Both are plain `transformers` checkpoints that use the CLS token,
  **not** sentence-transformers models: `SentenceTransformer(name)` silently
  falls back to mean pooling. Always build CLS pooling explicitly.
- Similarity search is exact, in numpy (no FAISS).
- The default suite must stay offline; anything that downloads a model gets
  `@pytest.mark.model`.
