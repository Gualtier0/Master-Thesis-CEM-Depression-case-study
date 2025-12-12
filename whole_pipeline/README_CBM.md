# Complete CBM Pipeline - Depression Detection

A clean, simple Concept Bottleneck Model (CBM) implementation for depression detection from social media posts.

## Overview

This pipeline implements a complete workflow for:
1. Loading preprocessed data from disk (saved by the CEM pipeline)
2. Training a simple Concept Bottleneck Model with 21 BDI-II depression concepts
3. Evaluating with comprehensive metrics and per-subject concept probabilities

## Differences from CEM Pipeline

### Architecture Comparison

**CBM (Concept Bottleneck Model):**
```
Input (384-dim) → Concept Extractor → Concept Logits (21)
                                      ↓ sigmoid
                                  Concept Probs (21)
                                      ↓
                                  Task Classifier → Task Prediction (1)
```

**CEM (Concept Embedding Model):**
```
Input (384-dim) → Pre-concept Model → Context Generators (21)
                                      ↓
                                  Concept Embeddings (21 × 128)
                                      ↓
                                  C2Y Model → Task Prediction (1)
```

### Key Differences

| Feature | CBM | CEM |
|---------|-----|-----|
| **Concept Representation** | Direct probabilities | Learned embeddings |
| **Architecture** | Simpler (2 stages) | More complex (multi-stage) |
| **Interpretability** | High (direct concept→task) | Moderate (embedding→task) |
| **Parameters** | Fewer (~200K) | More (~1.5M) |
| **Optimizer** | AdamW | SGD/Adam with LR scheduler |
| **Training** | Faster | Slower |
| **Intervention** | Easier (set concept probs) | Requires embedding intervention |

## Quick Start

1. **Ensure data is preprocessed:**
   ```bash
   # First run the CEM pipeline to generate preprocessed data
   jupyter notebook complete_cem_pipeline.ipynb
   # This creates data/processed/whole_pipeline/
   ```

2. **Navigate to the pipeline directory:**
   ```bash
   cd whole_pipeline
   ```

3. **Open the CBM notebook:**
   ```bash
   jupyter notebook complete_CBM_pipeline.ipynb
   ```

4. **Run all cells sequentially** (Cell → Run All)
   - The notebook will load preprocessed data and train automatically
   - Estimated runtime: 15-25 minutes on MacBook GPU (MPS)

## Requirements

- Python 3.8+
- PyTorch with MPS support (for MacBook GPU)
- pytorch-lightning
- scikit-learn
- pandas, numpy
- All dependencies from parent `requirements.txt`

## Data Requirements

The pipeline expects preprocessed data from the CEM pipeline:

```
data/processed/whole_pipeline/
├── train_data.npz          # X_train, C_train, y_train, subject_ids
├── val_data.npz            # X_val, C_val, y_val, subject_ids
├── test_data.npz           # X_test, C_test, y_test, subject_ids
└── class_weights.json      # Class imbalance information
```

## Output Files

After execution, the pipeline generates:

```
outputs_cbm/
├── models/
│   └── cbm-epoch=XX-val_loss=X.XX.ckpt     # Best model checkpoint
├── results/
│   ├── test_metrics.json                    # Evaluation metrics
│   └── test_predictions.csv                 # Per-subject predictions with concepts
└── logs/
    └── cbm_pipeline/                        # Training logs
```

## Pipeline Sections

### Section 0: Configuration & Setup
- Imports libraries and sets random seed
- Detects GPU (MPS/CUDA/CPU)
- Defines paths and hyperparameters
- Lists 21 BDI-II concept names

### Section 1: Load Preprocessed Data
- Loads train/val/test data from `data/processed/whole_pipeline/`
- Loads class weights for imbalanced training
- No data processing needed (already done by CEM pipeline)

### Section 2: PyTorch Dataset & DataLoaders
- Creates CBMDataset class
- Builds train/val/test loaders
- Batch size: 32 (train), 64 (val/test)

### Section 3: Concept Bottleneck Model
- Defines ConceptBottleneckModel class
- Architecture: Input(384) → FC(256) → ReLU → Dropout → Concepts(21)
                Concepts(21) → FC(64) → ReLU → Dropout → Task(1)
- Uses BCEWithLogitsLoss for both concept and task losses

### Section 4: Model Initialization
- Initializes ConceptBottleneckModel
- Applies class weights (pos_weight ~4.86) for imbalance

### Section 5: Training
- PyTorch Lightning trainer on MPS GPU
- 100 epochs with validation every epoch
- AdamW optimizer with weight decay
- ModelCheckpoint saves best model (lowest val_loss)

### Section 6: Test Evaluation
- Runs inference on test set
- Extracts concept probabilities and task predictions

### Section 7: Metrics & Results Display
- Computes comprehensive metrics:
  - Accuracy, Balanced Accuracy, ROC-AUC, MCC
  - F1-score (binary, macro, micro)
  - Precision, Recall
  - Confusion matrix
- Prints formatted results
- Saves metrics JSON and predictions CSV
- Shows concept activation statistics

### Section 8: Summary
- Displays output file locations

## Model Architecture

### Concept Extractor
- Input: 384-dim averaged SBERT embedding
- Hidden: 256-dim with ReLU + Dropout(0.3)
- Outputs 21 concept logits
- Sigmoid applied to get concept probabilities

### Task Classifier (C2Y)
- Input: 21 concept probabilities
- Hidden: 64-dim with ReLU + Dropout(0.2)
- Output: 1 task logit (depression prediction)

### Loss Function
```
Total Loss = Task Loss + Concept Loss Weight × Concept Loss
           = BCE(y_logits, y_true) + 1.0 × BCE(c_logits, c_true)
```

Note: Concept loss only computed when concept labels available (training set)

## Hyperparameters

```python
embedding_dim = 384               # SBERT dimension
n_concepts = 21                   # BDI-II concepts
batch_size_train = 32
batch_size_eval = 64
max_epochs = 100
learning_rate = 0.001             # Lower than CEM (AdamW)
weight_decay = 0.01
concept_loss_weight = 1.0
```

## Expected Performance

Based on typical CBM performance with class weights:

- **Accuracy:** ~85-90%
- **ROC-AUC:** ~0.85-0.90
- **F1 (Binary):** ~0.50-0.70
- **Balanced Accuracy:** ~0.70-0.75

Note: CBM typically has slightly lower performance than CEM but higher interpretability.

## Advantages of CBM

1. **Direct Interpretability:** Concepts are directly predicted probabilities, easy to understand
2. **Easier Intervention:** Can manually set concept values to test "what-if" scenarios
3. **Faster Training:** Simpler architecture trains faster
4. **Fewer Parameters:** Less prone to overfitting on small datasets
5. **Transparent Decision Path:** Clear X → C → Y reasoning chain

## Troubleshooting

### Out of Memory
- Reduce `batch_size_train` and `batch_size_eval`
- Reduce `max_epochs`

### MPS Not Available
- The notebook will automatically fall back to CPU
- Expect 2-3x longer runtime on CPU

### Missing Data Files
- Ensure you've run `complete_cem_pipeline.ipynb` first to generate preprocessed data
- Check that `data/processed/whole_pipeline/` contains all 4 files

### Poor Performance
- Try adjusting `learning_rate` (try 0.0001 or 0.01)
- Increase `concept_loss_weight` to prioritize concept learning
- Check class weights are being applied correctly

## Comparison with CEM

To compare CBM vs CEM performance:

1. Run both pipelines on the same data
2. Compare `outputs_cbm/results/test_metrics.json` with `outputs/results/test_metrics.json`
3. Key metrics to compare:
   - ROC-AUC (CEM typically 2-5% higher)
   - F1 Score (CEM typically 5-10% higher)
   - Balanced Accuracy (similar)
   - Concept prediction accuracy (CBM more direct)

## Citation

If you use this pipeline, please cite the original CBM paper:

```
@inproceedings{koh2020concept,
  title={Concept bottleneck models},
  author={Koh, Pang Wei and Nguyen, Thao and Tang, Yew Siang and Mussmann, Stephen and Pierson, Emma and Kim, Been and Liang, Percy},
  booktitle={International Conference on Machine Learning},
  pages={5338--5348},
  year={2020},
  organization={PMLR}
}
```

## License

This pipeline inherits the license from the parent project.

---

**Created:** 2025
**Maintainer:** Thesis Project
**Status:** Production Ready
