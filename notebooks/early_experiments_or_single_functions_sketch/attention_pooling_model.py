
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

def compute_accuracy(c_logits, y_logits, c, y):
    # Metrics for concepts
    try:
        c_probs = torch.sigmoid(c_logits)
        c_pred = (c_probs > 0.5).int()
        c_accuracy = torchmetrics.functional.accuracy(c_pred, c.int(), task="multilabel", num_labels=c.shape[-1])
        c_auc = torchmetrics.functional.auroc(c_probs, c.int(), task="multilabel", num_labels=c.shape[-1])
        c_f1 = torchmetrics.functional.f1_score(c_pred, c.int(), task="multilabel", num_labels=c.shape[-1])
        c_metrics = (c_accuracy, c_auc, c_f1)
    except (ValueError, RuntimeError) as e:
        c_metrics = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    # Metrics for task
    try:
        y_probs = torch.sigmoid(y_logits)
        y_pred = (y_probs > 0.5).int()
        y_squeezed = y.int().squeeze()
        y_accuracy = torchmetrics.functional.accuracy(y_pred.squeeze(), y_squeezed, task="binary")
        y_auc = torchmetrics.functional.auroc(y_probs.squeeze(), y_squeezed, task="binary")
        y_f1 = torchmetrics.functional.f1_score(y_pred.squeeze(), y_squeezed, task="binary")
        y_metrics = (y_accuracy, y_auc, y_f1)
    except (ValueError, RuntimeError) as e:
        y_metrics = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    return c_metrics, y_metrics


class AttentionPooler(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        attention_logits = self.attention_net(embeddings)
        attention_weights = F.softmax(attention_logits, dim=1)
        pooled_embedding = torch.sum(attention_weights * embeddings, dim=1)
        return pooled_embedding

class AttentionCBM(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        embedding_dim,
        c_extractor_arch,
        concept_loss_weight=1.0,
        task_loss_weight=1.0,
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=4e-05,
        momentum=0.9,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.attention_pooler = AttentionPooler(embedding_dim=embedding_dim)
        self.x_to_c_model = c_extractor_arch(output_dim=n_concepts)
        self.c_to_y_model = nn.Linear(n_concepts, n_tasks)

        self.loss_concept = nn.BCEWithLogitsLoss()
        self.loss_task = nn.BCEWithLogitsLoss()

    def forward(self, x):
        concept_logits = self.x_to_c_model(x)
        concept_probs = torch.sigmoid(concept_logits)
        task_logits = self.c_to_y_model(concept_probs)
        return concept_logits, task_logits

    def _unpack_batch(self, batch):
        x_raw, y, c = batch
        return x_raw, y, c

    def _run_step(self, batch, batch_idx):
        x_raw, y, c = self._unpack_batch(batch)
        
        pooled_embeddings = self.attention_pooler(x_raw)

        concept_logits, task_logits = self.forward(pooled_embeddings)
        
        concept_loss = self.loss_concept(concept_logits, c)
        task_loss = self.loss_task(task_logits.squeeze(), y.squeeze())
        
        loss = (
            self.hparams.concept_loss_weight * concept_loss +
            self.hparams.task_loss_weight * task_loss
        )
        
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
