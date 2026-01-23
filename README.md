# CEM-Concept embedded models for mental health screening

An application of CEM to eRisk benchmark dataset.

## Research questions:
- Are Concept-Based Models effective in mental healthcare settings despite the high level of label and textual noise?
- Does incorporating concept-level explainability necessarily entail a performance trade-off compared to end-to-end black-box models?
- Can a lightweight and easily deployable model achieve competitive performance for depression screening?

## Layout
- notebooks/Pipelines: the code.
- scripts: helper functions for quick tasks.
- data/raw/: original datasets, the datasets supporting this work are from eRisk collections and are available for research
purposes under signing user agreements. When comparing results with submissions of the shared task, identical train/test has been used.
- data/processed/: cleaned data/joined dataset.
- docs/: the actual thesis and pptx presentation for defence.

## Quickstart
1. Clone: git clone https://github.com/Gualtier0/Master-Thesis-CEM-Depression-etc-case-study.git
2. Install deps: pip install -r requirements.txt
3. Work: open notebooks/pipeline or run scripts

