
# Pipelines Notebooks

This folder contains the experimental pipelines used in the thesis.  
Notebooks are organized into two main groups:

- **Embedding and dataset preparation** (`0_*.ipynb`)
- **Model training and evaluation** (`1_*.ipynb`)

Most classifiers can be combined with different embedding pipelines, provided that the expected embedding dimensionality is respected. Each training notebook explicitly states its input requirements.

---

## Baseline Pipeline

The baseline results reported in the thesis are obtained using:

- **Dataset preparation:** `0c_prepare_max_alt_dataset.ipynb`
- **Model training:** `1d_CEM_max_Gold.ipynb`

This combination consistently provided the strongest performance across experiments.

The only configuration that achieved comparable or slightly improved results is:
- `1h_CEM_larger.ipynb`, which increases model capacity.

---

## Other Notebooks

All remaining pipelines and models are kept for completeness and reproducibility.  
They explore alternative aggregation strategies, architectural variations, loss functions, and stress-test settings that were investigated during the thesis but ultimately did not lead to improved performance. Some comments may refer to previous iteration of them and not be relevant anymore.

[User_Case_Study.ipynb](./User_Case_Study.ipynb) can be used to track users across the pipeline.

---

## Outputs

Folders prefixed with `outputs_` contain logs, checkpoints, and evaluation results produced by the corresponding notebooks.

