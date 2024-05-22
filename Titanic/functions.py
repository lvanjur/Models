import matplotlib.pyplot as plt
import os
import subprocess

from sklearn.metrics import (confusion_matrix, auc, f1_score, roc_curve,
                            precision_score, recall_score, accuracy_score)
from sklearn.inspection import permutation_importance


def evaluate_model(model, X_true, X_pred, y_true, y_pred,
                   write=True):

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision_sc = precision_score(y_true, y_pred)
    recall_sc = recall_score(y_true, y_pred)
    
    if write:
        write_model_metrics(accuracy, conf_matrix, f1, precision_sc, recall_sc)

    return
        
        
def write_model_metrics(accuracy,
                        confusion_matrix,
                        f1,
                        precision_score,
                        recall_score,
                        path='model_info.txt'):
    
    with open(path, 'w') as f:
        # Write model metrics to the file
        f.write('Model Metrics:\n')
        f.write('Accuracy: {}\n'.format(accuracy))
        f.write('Precision: {}\n'.format(precision_score))
        f.write('Recall: {}\n'.format(recall_score))
        f.write('F1 Score: {}\n'.format(f1))
        f.write('Confusion matrix: {}\n\n'.format(confusion_matrix))
            
    return


def get_feature_predictivity(model, X, y,
                             write=True,
                             path='model_info.txt'):
    
    feature_importances = permutation_importance(model, X, y,
                                                 n_repeats=10,
                                                 random_state=42)
    
    feature_importances_dict = dict(zip(model.feature_names_in_,
                                        feature_importances.importances_mean))
    
    if os.path.exists(path):
        mode = 'a'
    else:
        mode = 'w'
        
    with open(path, mode) as f:
        f.write('Model Feature Importances:\n')
        for feature, importance in feature_importances_dict.items():
            f.write(f'{feature}: {importance:.6f}\n')
    
    return


def create_evaluation_plots(model, X_true, X_pred, y_true, y_pred):
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_pred, model.predict_proba(X_pred)[:, 1])
    
    # Compute Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig('roc_plot.png', bbox_inches='tight')
    plt.show()

 
    return

