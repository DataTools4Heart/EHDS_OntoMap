# Contributing to ehds_ontomap

Thank you for your interest in improving `ehds_ontomap`. Everyone taking part
is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security issues privately as described in [SECURITY.md](SECURITY.md).

## Development setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nlp4bia-bsc/ehds_ontomap.git
cd ehds_ontomap
uv sync --group dev                     # lexical methods only
uv sync --group dev --extra biencoder   # include the bi-encoder method
uv run pre-commit install
```

## Checks

Every pull request must pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not model"
```

The default test suite is offline. Tests that download a Hugging Face model are
marked `@pytest.mark.model` and run with `uv run pytest -m model`.

## Guidelines

- Keep the public API typed and documented (NumPy-style docstrings).
- Every behaviour change needs a test and a line under `[Unreleased]` in
  `CHANGELOG.md`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Do not commit ontologies or terminology files. Many clinical terminologies
  (SNOMED CT, UMLS) are licensed and may not be redistributed.

## License

By contributing, you agree that your contributions are licensed under the
[EUPL-1.2](LICENSE).
