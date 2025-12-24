
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import sklearn

# This is a re-implementation of the metrics helper function as it could not be located.
# It uses torchmetrics, which is a dependency of the project.
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
        # Handle cases where a metric is not computable (e.g., only one class in batch)
        c_metrics = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    # Metrics for task
    try:
        y_probs = torch.sigmoid(y_logits)
        y_pred = (y_probs > 0.5).int()
        # Squeeze to handle (batch, 1) vs (batch,) shape issues
        y_squeezed = y.int().squeeze()
        y_accuracy = torchmetrics.functional.accuracy(y_pred.squeeze(), y_squeezed, task="binary")
        y_auc = torchmetrics.functional.auroc(y_probs.squeeze(), y_squeezed, task="binary")
        y_f1 = torchmetrics.functional.f1_score(y_pred.squeeze(), y_squeezed, task="binary")
        y_metrics = (y_accuracy, y_auc, y_f1)
    except (ValueError, RuntimeError) as e:
        y_metrics = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    return c_metrics, y_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification to address class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for positive class (0-1, typically class proportion)
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid), shape (N,) or (N, 1)
            targets: Ground truth labels, shape (N,) or (N, 1), values in {0, 1}
        """
        # Ensure compatible shapes
        logits = logits.view(-1)
        targets = targets.view(-1)

        # BCE loss component
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss for long-tailed recognition.

    Creates class-dependent margins to make decision boundaries harder for minority classes.
    For binary classification: minority class gets larger margin.

    Args:
        n_positive: Number of positive samples (83)
        n_negative: Number of negative samples (403)
        max_margin: Maximum margin value (default: 0.5, tune 0.3-1.0)
        scale: Temperature scaling (default: 30, tune 10-50)
    """
    def __init__(self, n_positive, n_negative, max_margin=0.5, scale=30):
        super(LDAMLoss, self).__init__()
        self.max_margin = max_margin
        self.scale = scale

        # Compute class frequencies
        total = n_positive + n_negative
        freq_pos = n_positive / total
        freq_neg = n_negative / total

        # Compute margins: minority class gets larger margin
        # Formula: margin = max_margin * (freq)^(-0.25)
        margin_pos = max_margin * (freq_pos ** (-0.25))
        margin_neg = max_margin * (freq_neg ** (-0.25))

        self.register_buffer('margin_pos', torch.tensor(margin_pos))
        self.register_buffer('margin_neg', torch.tensor(margin_neg))

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply class-dependent margins
        margin = targets * self.margin_pos + (1 - targets) * (-self.margin_neg)
        adjusted_logits = (logits - margin) * self.scale

        return F.binary_cross_entropy_with_logits(adjusted_logits, targets, reduction='mean')


# This is the new, patched model that combines ConceptEmbeddingModel and its parent,
# with the required fix for the loss function.
class PatchedConceptEmbeddingModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        input_dim, # Added for dummy forward pass
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,
        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=None, # Typically provided
        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,
        top_k_accuracy=None,
        use_focal_loss=False,
        focal_loss_alpha=None,
        focal_loss_gamma=2.0,
        use_ldam_loss=False,
        n_positive=None,
        n_negative=None,
        ldam_max_margin=0.5,
        ldam_scale=30,
    ):
        super().__init__()
        # We are doing a manual save of hyperparameters so we can
        # use them later
        self.save_hyperparameters()

        self.pre_concept_model = c_extractor_arch(output_dim=None)
        if self.hparams.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)


        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()

        # Dynamically get the output features of the extractor
        dummy_out_features = self._get_extractor_out_features(c_extractor_arch, input_dim)

        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(dummy_out_features, 2 * emb_size)
                    )
                )
            else:
                activation_layer = {
                    "leakyrelu": nn.LeakyReLU(),
                    "relu": nn.ReLU(),
                    "sigmoid": nn.Sigmoid(),
                }.get(embedding_activation)
                self.concept_context_generators.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(dummy_out_features, 2 * emb_size),
                        activation_layer
                    )
                )

            if self.hparams.shared_prob_gen and (len(self.concept_prob_generators) == 0):
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
            elif not self.hparams.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))

        if c2y_model is None:
            units = [n_concepts * emb_size] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        # ******************* THE FIX *******************
        # We use BCEWithLogitsLoss for numerical stability.
        self.loss_concept = torch.nn.BCEWithLogitsLoss(weight=weight_loss)
        # *********************************************

        # Task loss: LDAM > Focal > BCE priority
        if n_tasks > 1:
            self.loss_task = torch.nn.CrossEntropyLoss(weight=task_class_weights)
        else:
            if self.hparams.get('use_ldam_loss', False):
                self.loss_task = LDAMLoss(
                    n_positive=self.hparams.get('n_positive', 83),
                    n_negative=self.hparams.get('n_negative', 403),
                    max_margin=self.hparams.get('ldam_max_margin', 0.5),
                    scale=self.hparams.get('ldam_scale', 30)
                )
            elif self.hparams.use_focal_loss:
                self.loss_task = FocalLoss(
                    alpha=self.hparams.focal_loss_alpha,
                    gamma=self.hparams.focal_loss_gamma
                )
            else:
                self.loss_task = torch.nn.BCEWithLogitsLoss(pos_weight=task_class_weights)

    def _get_extractor_out_features(self, extractor_arch, input_dim):
        extractor = extractor_arch(output_dim=None)
        dummy_input = torch.randn(1, input_dim)
        dummy_output = extractor(dummy_input)
        return dummy_output.shape[-1]


    def _after_interventions(
        self,
        prob,
        intervention_idxs=None,
        c_true=None,
        train=False,
    ):
        if train and (self.hparams.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            mask = torch.bernoulli(
                self.ones * self.hparams.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor).to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
    ):
        pre_c = self.pre_concept_model(x)
        contexts = []
        c_logits_list = [] # Will store logits

        for i, context_gen in enumerate(self.concept_context_generators):
            if self.hparams.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
            context = context_gen(pre_c)
            # ******************* THE FIX *******************
            # Get the raw logit for the concept
            logit = prob_gen(context)
            c_logits_list.append(logit)
            # *********************************************
            contexts.append(torch.unsqueeze(context, dim=1))

        c_logits = torch.cat(c_logits_list, axis=-1)
        contexts = torch.cat(contexts, axis=1)

        # To get the probabilities for mixing, we pass logits through sigmoid
        c_probs = torch.sigmoid(c_logits)

        probs, intervention_idxs = self._after_interventions(
            c_probs, # Use probabilities for intervention mixing
            intervention_idxs=intervention_idxs,
            c_true=c,
            train=train,
        )
        c_pred_embs = (
            contexts[:, :, :self.hparams.emb_size] * torch.unsqueeze(probs, dim=-1) +
            contexts[:, :, self.hparams.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred_embs = c_pred_embs.view((-1, self.hparams.emb_size * self.hparams.n_concepts))
        y_logits = self.c2y_model(c_pred_embs)

        # ******************* THE FIX *******************
        # Return the raw concept LOGITS, the embeddings, and task logits
        return c_logits, c_pred_embs, y_logits
        # *********************************************

    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        intervention_idxs=None,
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            intervention_idxs=intervention_idxs,
        )

    def _unpack_batch(self, batch):
        x = batch[0]
        y, c = batch[1], batch[2]
        return x, y, c

    def _run_step(self, batch, batch_idx, train=False):
        x, y, c = self._unpack_batch(batch)
        c_logits, _, y_logits = self._forward(x, c=c, train=train)

        task_loss = self.loss_task(
            y_logits.squeeze(),
            y.squeeze(),
        )

        concept_loss = self.loss_concept(c_logits, c)
        loss = self.hparams.concept_loss_weight * concept_loss + self.hparams.task_loss_weight * task_loss


        # Compute accuracy and other metrics
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_logits, # Pass logits
            y_logits,
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
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log("train_" + name, val, prog_bar=("auc" in name))
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("auc" in name))

    def test_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)

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
