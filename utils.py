from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0


def draw_confusion_matrix(true, pred, dir, num_classes):
    cm = confusion_matrix(true, pred)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None], 
                index=list(range(num_classes)), columns=list(range(num_classes)))
    df = df.fillna(0)  # NaN 값을 0으로 변경

    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig(f"{dir}/confusion_matrix.png")
    plt.close('all')