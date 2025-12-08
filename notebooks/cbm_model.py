
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

# We reuse the compute_accuracy function from the patched_model.
# This avoids code duplication and ensures metrics are calculated consistently.
from patched_model import compute_accuracy

class SimpleCBM(pl.LightningModule):
    """
    A simple Concept Bottleneck Model (CBM).
    
    This model first maps the input 'x' to a set of concept predictions,
    and then maps the concept predictions to the final task label 'y'.
    """
    def __init__(
        self,
        n_concepts,
        n_tasks,
        input_dim,
        c_extractor_arch,
        concept_loss_weight=1.0,
        task_loss_weight=1.0,
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=4e-05,
        momentum=0.9,
    ):
        """
        :param int n_concepts: The number of concepts.
        :param int n_tasks: The number of output classes.
        :param int input_dim: The dimensionality of the input embeddings.
        :param Fun[(int), nn.Module] c_extractor_arch: A function that returns a Pytorch Module
                                                      for the concept extractor (x_to_c).
        :param float concept_loss_weight: Weight for the concept loss.
        :param float task_loss_weight: Weight for the task loss.
        """
        super().__init__()
        self.save_hyperparameters()

        # 1. x_to_c model: maps input x to concept logits
        self.x_to_c_model = c_extractor_arch(output_dim=n_concepts)
        
        # 2. c_to_y model: maps concept probabilities to task logits
        self.c_to_y_model = nn.Linear(n_concepts, n_tasks)

        # 3. Loss functions
        self.loss_concept = nn.BCEWithLogitsLoss()
        self.loss_task = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        Forward pass through the model.
        
        Returns a tuple of (concept_logits, task_logits).
        """
        # Get concept logits from the input
        concept_logits = self.x_to_c_model(x)
        
        # Get concept probabilities by applying a sigmoid
        # These probabilities are the "bottleneck"
        concept_probs = torch.sigmoid(concept_logits)
        
        # Get task logits from the concept probabilities
        task_logits = self.c_to_y_model(concept_probs)
        
        return concept_logits, task_logits

    def _unpack_batch(self, batch):
        x, y, c = batch
        return x, y, c

    def _run_step(self, batch, batch_idx):
        x, y, c = self._unpack_batch(batch)
        
        # Get model outputs
        concept_logits, task_logits = self.forward(x)
        
        # Calculate losses
        concept_loss = self.loss_concept(concept_logits, c)
        task_loss = self.loss_task(task_logits.squeeze(), y.squeeze())
        
        # Combine losses
        loss = (
            self.hparams.concept_loss_weight * concept_loss +
            self.hparams.task_loss_weight * task_loss
        )
        
        # Calculate metrics
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            concept_logits,
            task_logits,
            c,
            y,
        )
        
        result = {
            "c_accuracy": c_accuracy, "c_auc": c_auc, "c_f1": c_f1,
            "y_accuracy": y_accuracy, "y_auc": y_auc, "y_f1": y_f1,
            "concept_loss": concept_loss.detach(),
            "task_loss": task_loss.detach(),
            "loss": loss.detach(),
        }
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no)
        for name, val in result.items():
            self.log("train_" + name, val, prog_bar=("auc" in name))
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("auc" in name))

    def test_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no)
        for name, val in result.items():
            self.log("test_" + name, val)

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
