from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

def simple_accuracy(preds, labels):
    """Computes simple accuracy between predictions and labels."""
    return (preds == labels).mean()

def all_metrics(preds, labels):
    """Computes accuracy, precision, recall, and F1 score."""
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }

def compute_metrics(preds, labels):
    """Computes all metrics ensuring length consistency."""
    assert len(preds) == len(labels)
    return all_metrics(preds, labels)
