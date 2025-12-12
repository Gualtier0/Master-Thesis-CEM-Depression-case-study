# Complete CEM Pipeline - Concept Bottleneck (PyTorch Lightning)
# Minimal custom Concept Bottleneck Model implementation

import os
import glob
import re
import zipfile
import tempfile
import shutil
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer, util
from scipy.special import expit  # sigmoid

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
)

print("✓ All imports successful")

# -----------------------------
# Configuration (same as your notebook)
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED)

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data/raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data/processed")
OUTPUT_DIR = "outputs"

POS_DIR = os.path.join(DATA_RAW, "train/positive_examples_anonymous_chunks")
NEG_DIR = os.path.join(DATA_RAW, "train/negative_examples_anonymous_chunks")
TEST_DIR = os.path.join(DATA_RAW, "test")
TEST_LABELS = os.path.join(TEST_DIR, "test_golden_truth.txt")
CONCEPTS_FILE = os.path.join(DATA_PROCESSED, "merged_questionnaires.csv")

CONCEPT_NAMES = [
    "Sadness", "Pessimism", "Past failure", "Loss of pleasure",
    "Guilty feelings", "Punishment feelings", "Self-dislike", "Self-criticalness",
    "Suicidal thoughts or wishes", "Crying", "Agitation", "Loss of interest",
    "Indecisiveness", "Worthlessness", "Loss of energy", "Changes in sleeping pattern",
    "Irritability", "Changes in appetite", "Concentration difficulty",
    "Tiredness or fatigue", "Loss of interest in sex"
]
N_CONCEPTS = len(CONCEPT_NAMES)

HYPERPARAMS = {
    "k_posts": 20,
    "sbert_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "n_concepts": N_CONCEPTS,
    "n_tasks": 1,
    "emb_size": 128,
    "batch_size_train": 32,
    "batch_size_eval": 64,
    "max_epochs": 100,
    "learning_rate": 0.01,
    "weight_decay": 4e-05,
    "concept_loss_weight": 1.0,
}

print("✓ Hyperparameters configured")

# -----------------------------
# Data loading helpers (unchanged)
# -----------------------------
WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\u0000", "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_posts_from_xml(xml_path, min_chars=10):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"WARNING: Failed to parse {xml_path}: {e}")
        return []
    posts = []
    for writing in root.findall("WRITING"):
        title = writing.findtext("TITLE") or ""
        text = writing.findtext("TEXT") or ""
        combined = normalize_text(f"{title} {text}".strip())
        if len(combined) >= min_chars:
            posts.append(combined)
    return posts

# Note: for brevity, the rest of data loading, SBERT encoding and retrieval
# are assumed identical to your notebook. Paste them here when running.
# For this file we focus on the model implementation and wiring.

# -----------------------------
# Dataset and DataLoader (same as notebook)
# -----------------------------
class CEMDataset(Dataset):
    def __init__(self, X, C, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.C[idx]

# -----------------------------
# Minimal Concept Bottleneck Model (PyTorch Lightning)
# -----------------------------
class ConceptBottleneckModel(pl.LightningModule):
    """
    Minimal concept bottleneck model.
    Architecture:
      X -> concept extractor (logits) -> sigmoid -> predicted concepts
      predicted concepts -> simple classifier -> task logits
    Training loss:
      task_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
      concept_loss = BCEWithLogitsLoss (computed only when concept labels provided in batch)
      total_loss = task_loss + concept_loss_weight * concept_loss
    """
    def __init__(
        self,
        input_dim,
        n_concepts,
        task_output_dim=1,
        c_extractor_arch=None,
        learning_rate=1e-3,
        weight_decay=0.0,
        concept_loss_weight=1.0,
        pos_weight_tensor=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Concept extractor: if user provided a callable, use it to build final layer
        if c_extractor_arch is None:
            # simple two-layer MLP
            self.c_extractor = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_concepts),
            )
        else:
            # expect c_extractor_arch to be a callable that returns nn.Module when given output_dim
            self.c_extractor = c_extractor_arch(n_concepts)

        # classifier from predicted concepts to task logits
        self.c2y = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, task_output_dim)
        )

        # Loss functions
        # concept loss: per-concept BCE
        self.concept_loss_fn = nn.BCEWithLogitsLoss()
        # task loss: allow pos_weight for imbalanced binary classification
        if pos_weight_tensor is not None:
            self.task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(self.device))
        else:
            self.task_loss_fn = nn.BCEWithLogitsLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.concept_loss_weight = concept_loss_weight

    def forward(self, x):
        # x: (B, input_dim)
        c_logits = self.c_extractor(x)  # (B, n_concepts)
        # we'll produce y_logits from concept probabilities
        c_probs = torch.sigmoid(c_logits)
        y_logits = self.c2y(c_probs)  # (B, task_output_dim)
        return c_logits, y_logits

    def training_step(self, batch, batch_idx):
        x, y, c_true = batch
        c_logits, y_logits = self.forward(x)

        # task loss
        y = y.view_as(y_logits).float()
        task_loss = self.task_loss_fn(y_logits, y)

        # concept loss, only when concept ground truth labels exist in the batch
        concept_loss = torch.tensor(0.0, device=self.device)
        # detect if any label is provided (non all zeros and not all missing)
        if c_true.numel() > 0 and torch.sum(c_true) > 0:
            concept_loss = self.concept_loss_fn(c_logits, c_true)

        loss = task_loss + self.concept_loss_weight * concept_loss

        # logging
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_task_loss', task_loss, on_step=False, on_epoch=True)
        self.log('train_concept_loss', concept_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c_true = batch
        c_logits, y_logits = self.forward(x)

        y_prob = torch.sigmoid(y_logits).detach()
        y_pred = (y_prob >= 0.5).int()

        # compute losses similarly as training (concept loss only if labels present)
        y = y.view_as(y_logits).float()
        task_loss = self.task_loss_fn(y_logits, y)
        concept_loss = torch.tensor(0.0, device=self.device)
        if c_true.numel() > 0 and torch.sum(c_true) > 0:
            concept_loss = self.concept_loss_fn(c_logits, c_true)
        loss = task_loss + self.concept_loss_weight * concept_loss

        # return preds for metric aggregation
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {
            'val_loss': loss,
            'y_true': y.detach().cpu(),
            'y_prob': y_prob.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        x, y, c_true = batch
        c_logits, y_logits = self.forward(x)
        c_probs = torch.sigmoid(c_logits)
        y_probs = torch.sigmoid(y_logits).squeeze(-1)
        return {
            'y_true': y.detach().cpu(),
            'y_probs': y_probs.detach().cpu(),
            'c_probs': c_probs.detach().cpu(),
        }

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return opt

# -----------------------------
# Wiring: instantiate the model and plug into your existing training loop
# -----------------------------
# Example c_extractor_arch compatible with earlier code

def c_extractor_arch(output_dim):
    return nn.Sequential(
        nn.Linear(HYPERPARAMS['embedding_dim'], 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, output_dim)
    )

# When computing pos_weight tensor in your notebook, pass it here directly
# pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
# For demonstration we leave pos_weight None; pass the variable you computed in the notebook

# Instantiate the model
cem_model = ConceptBottleneckModel(
    input_dim=HYPERPARAMS['embedding_dim'],
    n_concepts=HYPERPARAMS['n_concepts'],
    task_output_dim=1,
    c_extractor_arch=c_extractor_arch,
    learning_rate=HYPERPARAMS['learning_rate'],
    weight_decay=HYPERPARAMS['weight_decay'],
    concept_loss_weight=HYPERPARAMS['concept_loss_weight'],
    pos_weight_tensor=None,  # replace with pos_weight_tensor computed earlier if desired
)

print("✓ ConceptBottleneckModel initialized")
print(cem_model)

# -----------------------------
# Training with PyTorch Lightning (same Trainer setup as your notebook)
# -----------------------------
# NOTE: make sure train_loader, val_loader are created as in your notebook
# Example trainer configuration shown below, use your existing callbacks/loggers

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=os.path.join(OUTPUT_DIR, "models"),
#     filename="cem-{epoch:02d}-{val_loss:.2f}",
#     save_top_k=1,
#     mode="min"
# )

# trainer = pl.Trainer(
#     max_epochs=HYPERPARAMS['max_epochs'],
#     accelerator=DEVICE,
#     devices=1,
#     logger=CSVLogger(save_dir=os.path.join(OUTPUT_DIR, "logs"), name="cem_pipeline"),
#     log_every_n_steps=10,
#     callbacks=[checkpoint_callback],
#     enable_progress_bar=True
# )

# trainer.fit(cem_model, train_loader, val_loader)

# -----------------------------
# Inference snippet (same semantics as your notebook)
# -----------------------------
# cem_model.eval()
# device_obj = torch.device(DEVICE)
# cem_model = cem_model.to(device_obj)
# with torch.no_grad():
#     for x_batch, y_batch, c_batch in test_loader:
#         x_batch = x_batch.to(device_obj)
#         c_logits, y_logits = cem_model(x_batch)
#         c_probs = torch.sigmoid(c_logits).cpu().numpy()
#         y_probs = torch.sigmoid(y_logits).cpu().squeeze().numpy()

# Save metrics/predictions as in your original notebook

print("✓ File ready. Paste your data-loading, loader creation and trainer.fit calls around this model implementation.")
