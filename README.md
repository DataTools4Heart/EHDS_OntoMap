# ehds_ontomap

[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](pyproject.toml)

**Map free-text clinical terms onto any `code | term` ontology.**

`ehds_ontomap` takes a list of terms, in whatever form you have it, and an
ontology stored as a two-column TSV or Parquet file. It returns ranked,
scored candidate codes for each term, using one of three methods:

| Method | What it does | Score |
|---|---|---|
| `string-match` | Exact match after normalisation | always `1.0` |
| `levenshtein` | Normalised Levenshtein similarity ([RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)) | `0` to `1` |
| `biencoder` | Cosine similarity of clinical term embeddings | `-1` to `1` |

It is developed within the [DataTools4Heart](https://www.datatools4heart.eu/)
(DT4H) project to support terminology harmonisation in the European Health
Data Space (EHDS).

> **Status:** alpha. The API may change before 1.0.

## Installation

`ehds_ontomap` is installed directly from GitHub; there is no need to clone
the repository. Python 3.10 or newer is required.

With **pip**:

```bash
# string-match and levenshtein
pip install "ehds_ontomap @ git+https://github.com/nlp4bia-bsc/ehds_ontomap.git"

# all three methods (adds PyTorch and Transformers)
pip install "ehds_ontomap[biencoder] @ git+https://github.com/nlp4bia-bsc/ehds_ontomap.git"
```

With **uv**, to add it to your project:

```bash
uv add "ehds_ontomap @ git+https://github.com/nlp4bia-bsc/ehds_ontomap.git"
uv add "ehds_ontomap[biencoder] @ git+https://github.com/nlp4bia-bsc/ehds_ontomap.git"
```

In a `requirements.txt` or `pyproject.toml` dependency list:

```text
ehds_ontomap[biencoder] @ git+https://github.com/nlp4bia-bsc/ehds_ontomap.git
```

To pin a release, append a tag or commit, e.g. `...ehds_ontomap.git@v0.1.0`.

## Quick start

```python
from ehds_ontomap import OntologyMapper

mapper = OntologyMapper("snomed_es.tsv")

terms = ["infarto de miocardio", "fallo cardiaco", "HTA"]
result = mapper.map(terms, method="biencoder", language="es", top_k=3)

result.best()  # one row per term with its top code
```

One-off calls can use `map_terms(terms, "snomed_es.tsv", method=...)`. Reuse
an `OntologyMapper` when mapping several lists: the ontology is prepared
(normalised or embedded) once per method.

## Inputs

**Ontology**: a TSV or Parquet file with two columns, `code` and `term`. A
code may have several terms (synonyms), and a term may belong to several codes
(homonyms). Codes are always read as text, so leading zeros are kept.

```text
code	term
22298006	infarto agudo de miocardio
22298006	IAM
84114007	insuficiencia cardiaca
```

**Terms**: any of the following:

- a Python list (or other iterable) of strings;
- a `dict` mapping your own IDs to terms;
- a pandas `Series` or `DataFrame` (`column=`, `id_column=`), or a Polars or
  PyArrow table;
- a path to a `.txt` (one term per line), `.csv`, `.tsv`, `.parquet`,
  `.json` or `.jsonl` file.

```python
mapper.map({"pt-001": "HTA", "pt-002": "asma"})
mapper.map("notes.csv", column="diagnosis", id_column="note_id")
```

## Methods

### `string-match` and `levenshtein`

Both compare normalised text: lowercased, with whitespace collapsed, accents
kept. Normalisation is configurable:

```python
from ehds_ontomap import Normalizer

mapper.map(terms, method="levenshtein", threshold=0.85, normalizer=Normalizer(strip_accents=True))
```

### `biencoder`

By default the model is chosen by `language`:

| Language | Default model |
|---|---|
| `es` | [ICB-UMA/ClinLinker-KB-P](https://huggingface.co/ICB-UMA/ClinLinker-KB-P) |
| `cs`, `en`, `it`, `nl`, `ro`, `sv` | [cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large](https://huggingface.co/cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large) |

Any other Hugging Face model can be used:

```python
mapper.map(terms, method="biencoder", model="my-org/my-encoder")
mapper.map(
    terms, method="biencoder", model="my-org/my-st-model", pooling=None
)  # sentence-transformers model
```

Other options: `pooling` (`"cls"` by default, `"mean"`, or `None` for a
sentence-transformers model's own configuration), `max_length` (25),
`batch_size`, `device`, `revision` and `cache_dir` (store the ontology
embeddings on disk so that later runs skip the encoding step).

Terms are embedded as written, without normalisation, because the models were
trained on cased text.

## Output

Every input term appears in the result, matched or not.

```python
result.to_frame()  # one row per (term, candidate)
result.best()  # one row per term: top candidate, n_candidates, ambiguous
result.summary()  # counts of matched, unmatched and ambiguous terms
result.unmatched  # the terms with no candidate
result.to_tsv("mapping.tsv")
result.to_parquet("mapping.parquet")
result.to_jsonl("mapping.jsonl")  # one line per term, candidates nested
```

`to_frame()` with the bi-encoder and `top_k=2`:

| query_id | query | rank | code | term | score | method |
|---|---|---|---|---|---|---|
| 0 | infarto de miocardio | 1 | 22298006 | infarto agudo de miocardio | 0.946 | biencoder |
| 0 | infarto de miocardio | 2 | 38341003 | hipertensión arterial | 0.468 | biencoder |
| 1 | HTA | 1 | 38341003 | hipertensión arterial | 0.904 | biencoder |
| 1 | HTA | 2 | 84114007 | insuficiencia cardiaca | 0.504 | biencoder |

A term with no candidate keeps one row with empty `rank`, `code`, `term` and
`score`.

### Thresholds, top-k and ties

| Parameter | Default | Meaning |
|---|---|---|
| `top_k` | `1` | Maximum number of distinct codes per term (`None` for all). |
| `threshold` | `None` | Minimum score, inclusive. Scores are method-specific. |
| `ties` | `"keep"` | `"keep"` also returns candidates tied with the last place; `"cut"` returns exactly `top_k`. |

Each code appears at most once per term, with its best-scoring synonym. Ranks
use competition ranking (1, 1, 3), so ties stay visible. A term that is a
homonym of several codes in the ontology therefore returns all of them, and
`best()` flags it as `ambiguous`.

Bi-encoder scores are not probabilities. Inspect the score distribution on
your own data before choosing a threshold for it.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Please follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and report security issues as described
in [SECURITY.md](SECURITY.md).

## Citation

If you use `ehds_ontomap` in your work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## Contact

- Mohamed Aziz Ouhida — [mohamed.ouhida@bsc.es](mailto:mohamed.ouhida@bsc.es)
- Fernando Gallego — [fernando.gallego@bsc.es](mailto:fernando.gallego@bsc.es)

## License

Licensed under the [European Union Public Licence v. 1.2](LICENSE) (EUPL-1.2).

Parts of the matching code are adapted from
[dt4h-annotation-api](https://github.com/nlp4bia-bsc/dt4h-annotation-api)
(MIT License, Copyright (c) 2026 NLP4BIA). The bi-encoder models are
distributed by their authors under their own licences.

## Acknowledgements

Developed by the [NLP4BIA](https://github.com/nlp4bia-bsc) group at the
[Barcelona Supercomputing Center](https://www.bsc.es/).

This project has received funding from the European Union's Horizon Europe
research and innovation programme under grant agreement No 101057849
(DataTools4Heart).
