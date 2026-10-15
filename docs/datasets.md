# Datasets

This document records, for each dataset used by RADIANT, its name,
source, license, permitted use, attribution requirements, and
redistribution restrictions.

Rule: if a license cannot be verified against the official source,
the entry is marked `LICENSE STATUS: UNVERIFIED` and must be
resolved before any public release.

## NIH ChestX-ray14 (vision)

- **Name:** NIH ChestX-ray14 (also known as ChestX-ray14)
- **Source URL:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Publisher:** National Institutes of Health Clinical Center
- **Format:** PNG images, 112,120 frontal chest X-rays from 30,805 patients, 14 finding labels
- **Local path:** `D:\NIH-ChestXray14\images\` (112,120 PNGs, ~43 GB)
- **Metadata files:** `D:\Data_Entry_2017.csv`, `D:\train_val_list.txt`, `D:\test_list.txt`
- **License:** CC0 1.0 Universal (Public Domain Dedication). Copy, modify, distribute, and use — including commercially — without permission.
- **Attribution:** Not required under CC0. Citation appreciated.
- **RADIANT usage:** Visual model evaluation (Stage 14) and real-image inference smoke tests (Stage 4).
- **Redistribution:** Permitted under CC0, but images are git-ignored to keep the repository small.
- **Public / restricted:** Publicly downloadable.

**Note on this installation:** the dataset is present on a local drive outside the repository
(`D:\`), not inside `radiant\data\nih\`. Scripts that need it read the path from
`config.py` (`NIH_IMAGES_DIR`), which defaults to `D:\NIH-ChestXray14\images`.
## Rad-SpRL (report NLP)

- **Name:** Rad-SpRL
- **Source URL:** https://data.mendeley.com/datasets/yhb26hfz8n/1
- **Publisher:** Mendeley Data
- **Format:** `NOT VERIFIED — annotation format must be inspected once the files are present. Do not assume a schema.`
- **License:** `LICENSE STATUS: UNVERIFIED — must be verified against the Mendeley Data page before use.`
- **Attribution:** Required per Mendeley Data terms.
- **RADIANT usage:** Evaluation of the deterministic NLP extractor (Stage 5). Rad-SpRL annotations are not used as ground truth for the vision model.
- **Redistribution:** Not redistributed. `data/radsprl/` is git-ignored.
- **Public / restricted:** Publicly downloadable from the Mendeley Data page.

**Mapping to NIH labels:** `NOT DOCUMENTED — inspection pending dataset availability.` Rad-SpRL may contain labels with no direct NIH ChestX-ray14 equivalent; any such labels will be listed explicitly in a future revision of this document.

## Synthetic data (authored for RADIANT)

- **Location:** `data/synthetic/`, `data/sample_xrays/`
- **Authorship:** Authored for this project. Not derived from any real patient or real report.
- **License:** Project's own license applies (see `LICENSE`, added in a later stage).
- **Attribution:** Not required.
- **RADIANT usage:** Unit tests, integration tests, and demo scenarios.
- **Redistribution:** May be redistributed with the project.

**Labelling requirement:** every synthetic file carries an explicit
`SYNTHETIC` marker. Synthetic reports begin and end with
`SYNTHETIC REPORT — NOT A REAL PATIENT DOCUMENT`. Synthetic images
are named with a `SYNTHETIC_` prefix. The UI must label any synthetic
input as such.

## Development fixtures (RAG knowledge base)

Development fixtures for the evidence RAG live in `data/knowledge_base/`
and are documented in the RAG stage. Each fixture starts with
`DEVELOPMENT FIXTURE — NOT AUTHORITATIVE CLINICAL GUIDANCE` and carries
`source:` and `license:` lines. They are test infrastructure only and
must never be presented in the UI as clinical guidance.

## What is NOT used

The following datasets are explicitly forbidden by project rules and
must never be added to this repository:

- MIMIC-IV, MIMIC-CXR
- CSRRG
- Credentialed PhysioNet data
- CheXpert Plus if restricted
- CANDID-PTX, CANDID-II
- ReXGradient-160K
- Any DUA-restricted or credentialed dataset

Do not propose replacement datasets. If a locked dataset is
unavailable, build the loader and interface and mark real-data
execution `NOT RUN`.