import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score,r2_score,matthews_corrcoef,confusion_matrix,average_precision_score

def analysis(y_true, y_pred):
    binary_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    binary_true = [1 if true >= 0.5 else 0 for true in y_true]

    # continous evaluate
    r2 = r2_score(y_true, y_pred)

    # binary evaluate
    binary_acc = accuracy_score(binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred)
    recall = recall_score(binary_true, binary_pred)
    f1 = f1_score(binary_true, binary_pred)
    # y_pred evaluate
    roauc = roc_auc_score(binary_true, y_pred)
    prauc = average_precision_score(binary_true, y_pred)
    
    mcc = matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'r2': r2,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roauc': roauc,
        'prauc':prauc,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result
