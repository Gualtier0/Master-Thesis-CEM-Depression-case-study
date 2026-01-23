# Concept Embedding Models for Mental Health Screening

This repository contains the code, experiments, and documentation developed for a Master’s thesis on **concept-based modeling for mental health screening**, with a focus on **depression detection** using the **eRisk benchmark dataset**.

The work investigates whether **Concept Embedding Models (CEMs)** can provide meaningful interpretability without sacrificing predictive performance in a highly noisy, weakly supervised clinical NLP setting.

---

## Problem Setting

Mental health screening from social media data presents several challenges:

- Labels are assigned at the user level, often based on self-reported diagnoses or strong indications, rather than clinical assessments.
- Textual data is noisy, informal, and highly variable across users and time.
- The task is strongly imbalanced and recall-sensitive, reflecting real-world screening requirements.
- Some users exhibit very few or no detectable linguistic signals, suggesting an intrinsic upper bound on performance.

The eRisk dataset provides a realistic benchmark to study these challenges in a controlled yet demanding environment.

---

## Approach

This work proposes and evaluates a **concept-based pipeline** built around Concept Embedding Models, where:

- Clinical concepts are defined according to **BDI-II symptom dimensions**.
- Sentence-level embeddings are retrieved using a **max-based similarity strategy** to capture extreme signals.
- Subject-level representations are formed through attention pooling.
- A CEM jointly predicts concept activations and the final depression risk, enabling post-hoc and intervention-based interpretability.

The pipeline is designed to be **lightweight, modular, and easily deployable**, avoiding heavy end-to-end fine-tuning while retaining competitive performance.

---

## Research Questions

- Are Concept-Based Models effective in mental healthcare settings despite the high level of label and textual noise?
- Does incorporating concept-level explainability necessarily entail a performance trade-off compared to end-to-end black-box models?
- Can a lightweight and easily deployable model achieve competitive performance for depression screening?

---

## Repository Structure
| Folder | Description |
|---------|--------------|
| [notebooks/Pipelines/](./notebooks/Pipelines/) | Core experimental pipelines and model training notebooks |
| [scripts/](./scripts/) | Helper scripts and utilities |
| [earliest_experiments/](./earliest_experiments/) | Preliminary and exploratory experiments |
| [logs/](./logs/) |Training logs and checkpoints |
| [docs/](./docs/) |Thesis document and defense presentation |
| [data/raw/](./data/raw/) | Original eRisk datasets (available under user agreements) |
| [data/processed/](./data/processed/) | Cleaned and joined datasets used in experiments |

---

## Reproducibility

All experiments are implemented in Python using standard scientific and deep learning libraries.

- Exact library versions are specified in `requirements.txt`
- Random seeds are fixed where applicable
- No proprietary software is required
- The full preprocessing, training, validation, and evaluation pipeline is included
- When comparing results with official eRisk submissions, identical train, validation, and test splits are used.

---

## Quick Start

```bash
git clone https://github.com/Gualtier0/Master-Thesis-CEM-Depression-etc-case-study.git
cd Master-Thesis-CEM-Depression-etc-case-study
pip install -r requirements.txt
```
Open the notebooks in notebooks/Pipelines/ to reproduce the main experiments, or run the provided scripts for specific tasks.

---

## Notes

Due to dataset licensing constraints, raw eRisk data is not redistributed. Users must obtain access independently and place the data in the expected directory structure.

---

## Citation

If you use or build upon this work, please cite the corresponding Master’s thesis.





