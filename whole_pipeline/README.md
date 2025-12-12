# Complete CEM Pipeline - Depression Detection

A clean, end-to-end pipeline for training and evaluating Concept Embedding Models (CEM) on depression detection from social media posts.

## Overview

This pipeline implements a complete workflow for:
1. Loading training/test data from XML files
2. Retrieving top-20 concept-relevant posts per subject using semantic similarity
3. Averaging SBERT embeddings into single vectors per subject
4. Training a Concept Embedding Model with 21 BDI-II depression concepts
5. Evaluating with comprehensive metrics and per-subject concept probabilities

## Quick Start

1. **Navigate to the pipeline directory:**
   ```bash
   cd whole_pipeline
   ```

2. **Open the notebook:**
   ```bash
   jupyter notebook complete_cem_pipeline.ipynb
   ```

3. **Run all cells sequentially** (Cell → Run All)
   - The notebook will execute the entire pipeline automatically
   - Estimated runtime: 30-45 minutes on MacBook GPU (MPS)

## Requirements

- Python 3.8+
- PyTorch with MPS support (for MacBook GPU)
- sentence-transformers
- pytorch-lightning
- scikit-learn
- pandas, numpy
- All dependencies from parent `requirements.txt`

## Data Requirements

The pipeline expects data in the following structure (relative to project root):

```
data/
├── raw/
│   ├── train/
│   │   ├── positive_examples_anonymous_chunks/  # Training positive cases
│   │   └── negative_examples_anonymous_chunks/  # Training negative cases
│   └── test/
│       ├── chunk 1.zip through chunk 10.zip     # Test data (zipped)
│       └── test_golden_truth.txt                # Test labels
└── processed/
    └── merged_questionnaires.csv                # Concept labels for training
```

## Output Files

After execution, the pipeline generates:

```
outputs/
├── models/
│   └── cem-epoch=XX-val_loss=X.XX.ckpt     # Best model checkpoint
├── results/
│   ├── test_metrics.json                    # Evaluation metrics
│   └── test_predictions.csv                 # Per-subject predictions with concepts
└── logs/
    └── cem_pipeline/                        # Training logs
```

## Pipeline Sections

### Section 0: Configuration & Setup
- Imports libraries and sets random seed
- Detects GPU (MPS/CUDA/CPU)
- Defines paths and hyperparameters
- Lists 21 BDI-II concept names

### Section 1: Load Training Data
- Parses XML files from positive/negative directories
- Extracts ~295k posts from 486 training subjects
- Loads concept labels from questionnaires

### Section 2: Load Test Data
- Extracts test ZIP files
- Parses 401 test subjects
- Splits 50/50 into validation (200) and test (201) sets

### Section 3: SBERT Setup
- Loads `all-MiniLM-L6-v2` model on GPU
- Creates embeddings for 21 BDI-II concepts

### Section 4: Post Retrieval
- For each subject: encodes all posts
- Computes semantic similarity with concepts
- Selects top-20 most relevant posts

### Section 5: Embedding Aggregation
- Encodes selected 20 posts per subject
- Averages embeddings → single 384-dim vector
- Builds concept matrices (C) and labels (y)

### Section 6: PyTorch Dataset & DataLoaders
- Creates CEMDataset class
- Builds train/val/test loaders
- Batch size: 32 (train), 64 (val/test)

### Section 7: CEM Model
- Initializes PatchedConceptEmbeddingModel
- Architecture: Input(384) → FC(256) → ReLU → Dropout → Concepts(21) → Task(1)
- Uses BCEWithLogitsLoss for both concept and task losses

### Section 8: Training
- PyTorch Lightning trainer on MPS GPU
- 100 epochs with validation every epoch
- ModelCheckpoint saves best model (lowest val_loss)

### Section 9: Test Evaluation
- Runs inference on test set
- Extracts concept probabilities and task predictions

### Section 10: Results Display
- Computes comprehensive metrics:
  - Accuracy, Balanced Accuracy, ROC-AUC, MCC
  - F1-score (binary, macro, micro)
  - Precision, Recall
  - Confusion matrix
- Prints formatted results
- Saves metrics JSON and predictions CSV
- Shows concept activation statistics

### Section 11: Cleanup
- Removes temporary test data directory

## Key Features

### 1. **Single Notebook Execution**
Run the entire pipeline without manual intervention.

### 2. **GPU Acceleration**
Automatically uses MacBook GPU (MPS) when available, falls back to CUDA or CPU.

### 3. **Concept-Based Post Selection**
Uses semantic similarity to select the most relevant posts for each subject, improving signal-to-noise ratio.

### 4. **Comprehensive Evaluation**
Provides detailed metrics and per-subject concept probabilities for interpretability.

### 5. **Proper Data Splits**
- Training: 486 subjects (83 positive, 403 negative)
- Validation: 200 subjects (stratified)
- Test: 201 subjects (stratified)

## Model Architecture

### Concept Extractor
- Input: 384-dim averaged SBERT embedding
- Hidden: 256-dim with ReLU + Dropout(0.3)
- Outputs 21 concept logits

### CEM Components
- **Pre-concept model:** Feature extraction
- **Concept context generators:** 21 context embeddings (2×128 dims each)
- **Concept probability generators:** Shared across concepts
- **C2Y model:** Concept embeddings → Task prediction

### Loss Function
```
Total Loss = Concept Loss + Task Loss
           = BCE(c_logits, c_true) + BCE(y_logits, y_true)
```

## Hyperparameters

```python
k_posts = 20                      # Top posts per subject
sbert_model = "all-MiniLM-L6-v2" # Embedding model
embedding_dim = 384               # SBERT dimension
n_concepts = 21                   # BDI-II concepts
emb_size = 128                    # Concept embedding size
batch_size_train = 32
batch_size_eval = 64
max_epochs = 100
learning_rate = 0.01
concept_loss_weight = 1.0
training_intervention_prob = 0.25
```

## Expected Performance

Based on the original implementation, expected test set metrics:
- **Accuracy:** ~92-93%
- **ROC-AUC:** ~0.96
- **F1 (Binary):** ~0.78-0.80
- **MCC:** ~0.74-0.75

## Troubleshooting

### Out of Memory
- Reduce `batch_size_train` and `batch_size_eval`
- Reduce `max_epochs`

### MPS Not Available
- The notebook will automatically fall back to CPU
- Expect 3-5x longer runtime on CPU

### Missing Data Files
- Ensure all data directories exist relative to project root
- Check that test ZIP files are present in `data/raw/test/`

### Import Errors
- Verify `patched_model.py` is in the `whole_pipeline/` directory
- Install missing dependencies: `pip install -r ../requirements.txt`

## Differences from Original Notebooks

1. **Single file:** All steps in one notebook vs. 11+ separate notebooks
2. **Simple averaging:** Uses mean of embeddings instead of 3-model ensemble
3. **Clean code:** Removed debug code, simplified logic
4. **Automatic splits:** No manual split configuration needed
5. **Better documentation:** Detailed markdown sections and comments

## Citation

If you use this pipeline, please cite the original CEM paper:
```
@article{zarlenga2022concept,
  title={Concept Embedding Models},
  author={Zarlenga, Mateo and Barbiero, Pietro and Ciravegna, Gabriele and others},
  journal={arXiv preprint arXiv:2209.09056},
  year={2022}
}
```

## License

This pipeline inherits the license from the parent project.

---

**Created:** 2025
**Maintainer:** Thesis Project
**Status:** Production Ready
