"""
Threshold Optimization Script
Find the best decision threshold to maximize recall while maintaining acceptable precision
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Load predictions from the CEM pipeline
predictions = pd.read_csv('outputs/results/test_predictions.csv')

y_true = predictions['y_true'].values
y_prob = predictions['y_prob'].values

print("="*70)
print("THRESHOLD OPTIMIZATION FOR MINORITY CLASS")
print("="*70)
print(f"\nCurrent performance with threshold=0.5:")
y_pred_05 = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred_05)
tn, fp, fn, tp = cm.ravel()
print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
print(f"  Recall: {recall_score(y_true, y_pred_05):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred_05):.4f}")
print(f"  F1: {f1_score(y_true, y_pred_05):.4f}")

print(f"\n{'Threshold':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Recall':<10} {'Precision':<10} {'F1':<10}")
print("-"*70)

best_f1 = 0
best_threshold = 0.5
best_recall = 0

# Try different thresholds
for threshold in np.arange(0.05, 0.95, 0.05):
    y_pred = (y_prob >= threshold).astype(int)

    if np.sum(y_pred) == 0:  # Skip if no positives predicted
        continue

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred) if tp + fp > 0 else 0
    f1 = f1_score(y_true, y_pred)

    print(f"{threshold:<12.2f} {tp:<6} {fp:<6} {fn:<6} {recall:<10.4f} {precision:<10.4f} {f1:<10.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_recall = recall

print("\n" + "="*70)
print(f"RECOMMENDED THRESHOLD: {best_threshold:.2f}")
print(f"  This achieves F1={best_f1:.4f}, Recall={best_recall:.4f}")
print("="*70)

# Show detailed results for recommended threshold
y_pred_best = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print(f"\nDetailed metrics with threshold={best_threshold:.2f}:")
print(f"  True Positives:  {tp}/{int(np.sum(y_true))} ({100*tp/np.sum(y_true):.1f}% of actual positives caught)")
print(f"  False Positives: {fp}/{int(len(y_true) - np.sum(y_true))} ({100*fp/(len(y_true)-np.sum(y_true)):.1f}% false alarm rate)")
print(f"  False Negatives: {fn}/{int(np.sum(y_true))} (missed {fn} depression cases)")
print(f"  Recall:          {recall_score(y_true, y_pred_best):.4f}")
print(f"  Precision:       {precision_score(y_true, y_pred_best):.4f}")
print(f"  F1 Score:        {f1_score(y_true, y_pred_best):.4f}")

# Find threshold for 85% recall target
print("\n" + "="*70)
print("FINDING THRESHOLD FOR 85% RECALL TARGET:")
print("="*70)

for threshold in np.arange(0.01, 1.0, 0.01):
    y_pred = (y_prob >= threshold).astype(int)

    if np.sum(y_pred) == 0:
        continue

    recall = recall_score(y_true, y_pred)

    if recall >= 0.85:
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Recall:    {recall:.4f} ✓")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        break
else:
    print("\n⚠ Cannot achieve 85% recall with current model predictions!")
    print("   The model probabilities are too low.")
    print("   You need to:")
    print("   1. Enable Focal Loss")
    print("   2. Use more aggressive class weights (10x-15x)")
    print("   3. Retrain the model")
