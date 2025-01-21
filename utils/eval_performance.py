import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score


def evaluate(model, data, batch_size, device):
    model.eval()
    total_preds, total_targets = [], []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            c1, c2, target = zip(*batch)
            c1 = torch.tensor(c1).to(device)
            c2 = torch.tensor(c2).to(device)
            target = torch.tensor(target).to(device).float()

            logit_H, logit_N, logit_T = model(c1, c2)    # Get model logits

            y_H = torch.sigmoid(logit_H)
            y_N = torch.sigmoid(logit_N)
            y_T = torch.sigmoid(logit_T)

            p_mean = (y_H + y_N + y_T) / 3.0    # Average the outputs
            predictions = p_mean.cpu().numpy()
            targets = target.cpu().numpy()

            total_preds.extend(predictions)
            total_targets.extend(targets)

    total_preds = np.array(total_preds).flatten()
    total_targets = np.array(total_targets).flatten()

    pred_label = (total_preds > 0.5).astype(int)    # Binarize predictions
    target_label = total_targets.astype(int)

    metrics = {
        "ACC": accuracy_score(target_label, pred_label),
        "F1": f1_score(target_label, pred_label),
        "Precision": precision_score(target_label, pred_label, zero_division=1),
        "Recall": recall_score(target_label, pred_label),
        "AUC": roc_auc_score(target_label, total_preds),
        "AP": average_precision_score(target_label, total_preds)
    }

    return metrics


def evaluate_test(model, data, batch_size, device, save_path):
    model.eval()
    total_preds, total_targets, total_c1, total_c2 = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            c1, c2, target = zip(*batch)
            c1 = torch.tensor(c1).to(device)
            c2 = torch.tensor(c2).to(device)
            target = torch.tensor(target).to(device).float()

            logit_H, logit_N, logit_T = model(c1, c2)    # Get model logits
            y_H = torch.sigmoid(logit_H)
            y_N = torch.sigmoid(logit_N)
            y_T = torch.sigmoid(logit_T)

            p_mean = (y_H + y_N + y_T) / 3.0    # Average the outputs
            predictions = p_mean.cpu().numpy()
            targets = target.cpu().numpy()

            total_preds.extend(predictions)
            total_targets.extend(targets)
            total_c1.extend(c1.cpu().numpy())
            total_c2.extend(c2.cpu().numpy())

    total_preds = np.array(total_preds).flatten()
    total_targets = np.array(total_targets).flatten()

    pred_label = (total_preds > 0.5).astype(int)    # Binarize predictions
    target_label = total_targets.astype(int)

    if save_path:
        results_df = pd.DataFrame({
            'Concept_1': total_c1,
            'Concept_2': total_c2,
            'Predictions': pred_label,
            'Targets': target_label
        })
        results_df.to_csv(save_path, index=False)

    metrics = {
        "ACC": accuracy_score(target_label, pred_label),
        "F1": f1_score(target_label, pred_label),
        "Precision": precision_score(target_label, pred_label, zero_division=1),
        "Recall": recall_score(target_label, pred_label),
        "AUC": roc_auc_score(target_label, total_preds),
        "AP": average_precision_score(target_label, total_preds)
    }

    return metrics
