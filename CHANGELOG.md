# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ontology loading from TSV and Parquet (`code | term`) with validation.
- Flexible term input: lists, mappings, pandas Series/DataFrame, Polars/PyArrow
  tables, and `.txt`, `.csv`, `.tsv`, `.parquet`, `.json`, `.jsonl` files.
- `MappingResult` with long, best-match, TSV, Parquet and JSONL views.
- Shared candidate selection: inclusive threshold, `top_k`, per-code
  deduplication, tie handling and competition ranking.
- Text normalisation (NFC, whitespace, optional case/accents/punctuation).
- `OntologyMapper` and `map_terms()` entry points.
- Methods: `string-match` (returns every homonym), `levenshtein` (RapidFuzz
  normalised similarity) and `biencoder` (ClinLinker-KB-P for Spanish, SapBERT
  XLM-R large for the other languages, or any Hugging Face model; CLS pooling;
  optional on-disk embedding cache).
- Exact, memory-bounded top-k search over the full ontology.
- Code of Conduct, security policy and citation metadata.
