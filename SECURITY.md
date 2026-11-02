# Security Policy

## Supported versions

`ehds_ontomap` is in alpha. Security fixes are applied to the latest release
and to the `main` branch only.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report them privately by email to the maintainers:

- Mohamed Ouhida — [mohamed.ouhida@bsc.es](mailto:mohamed.ouhida@bsc.es)
- Fernando Gallego — [fernando.gallego@bsc.es](mailto:fernando.gallego@bsc.es)

Include a description of the issue, the steps to reproduce it and the affected
version. We aim to acknowledge reports within five working days.

## Data protection

`ehds_ontomap` runs entirely locally and sends no data anywhere. The only
network access is the one-time download of bi-encoder models from the Hugging
Face Hub. Do not attach real patient data or licensed terminology content
(SNOMED CT, UMLS) to issues or reports.
