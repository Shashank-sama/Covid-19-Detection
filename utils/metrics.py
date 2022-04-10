import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def metrics(Y_true,predictions):
    """
    Function to print the performance metrics.

    Inputs
    Y_true: Ground truth labels
    predictions: Predicted labels

    Outputs
    Accuracy, F1 Score, Recall, Precision, Classification report, Confusion matrix
    """

    print('Accuracy:', accuracy_score(Y_true, predictions))
    print('F1 score:', f1_score(Y_true, predictions,average='weighted'))
    print('Recall:', recall_score(Y_true, predictions,average='weighted'))
    print('Precision:', precision_score(Y_true, predictions, average='weighted'))
    print('\n Clasification report:\n', classification_report(Y_true, predictions))
    print('\n Confusion matrix:\n',confusion_matrix(Y_true, predictions))

    #Creating confussion matrix
    snn_cm = confusion_matrix(Y_true, predictions)
    # Plotting cofusion matrix
    snn_df_cm = pd.DataFrame(snn_cm, range(2), range(2))  
    plt.figure(figsize = (9,5))  
    sn.set(font_scale=1.4)
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}, fmt="d")  
    plt.show()  



def get_true_pos(y, pred, th=0.5):

    """
    Function to calculate the total of true positive (TP) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))



def get_true_neg(y, pred, th=0.5):

    """
    Function to calculate the total of true negative (TN) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))



def get_false_neg(y, pred, th=0.5):

    """
    Function to calculate the total of false negative (FN) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """
    
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))



def get_false_pos(y, pred, th=0.5):

    """
    Function to calculate the total of false positive (FP) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))



def print_confidence_intervals(class_labels, statistics):

    """
    Function to calculate the confidence interval (5%-95%).
    
    Inputs
    class_labels: List with class names
    statistics: 

    Outputs
    Returns DataFrame with confidence intervals for each class
    """

    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df



def get_roc_curve(gt, pred, target_names):

    """
    Function to plot the ROC curve.
    
    Inputs
    gt: Ground truth labels
    pred: Predicted labels
    target_names: List with class names 
    """

    for i in range(len(target_names)):
        auc_roc = roc_auc_score(gt[:, i], pred[:, i])
        label = target_names[i] + " AUC: %.3f " % auc_roc
        xlabel = "False positive rate"
        ylabel = "True positive rate"
        a, b, _ = roc_curve(gt[:, i], pred[:, i])
        plt.figure(1, figsize=(7, 7))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(a, b, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)



def get_prc_curve(gt, pred, target_names):

    """
    Function to plot the Precision-Recall curve.
    
    Inputs
    gt: Ground truth labels
    pred: Predicted labels
    target_names: List with class names 
    """

    for i in range(len(target_names)):
        precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
        average_precision = average_precision_score(gt[:, i], pred[:, i])
        label = target_names[i] + " Avg.: %.3f " % average_precision
        plt.figure(1, figsize=(7, 7))
        plt.step(recall, precision, where='post', label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)



def plot_calibration_curve(y, pred,class_labels):

    """
    Function to plot the calibration curve.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    class_labels: List with class names 
    """

    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()



def get_accuracy(y, pred, th=0.5):

    """
    Function to calculate the accuracy.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    accuracy = 0.0
    TP = get_true_pos(y, pred, th)
    FP = get_false_pos(y, pred, th)
    TN = get_true_neg(y, pred, th)
    FN = get_false_pos(y, pred, th)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return accuracy



def get_sensitivity(y, pred, th=0.5):

    """
    Function to calculate the sensitivity.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    sensitivity = 0.0
    TP = get_true_pos(y,pred,th)
    FN = get_false_neg(y,pred,th)
    sensitivity = TP/(TP+FN)
    return sensitivity



def get_specificity(y, pred, th=0.5):

    """
    Function to calculate the specificity.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """
    
    specificity = 0.0
    TN = get_true_neg(y,pred,th)
    FP = get_false_pos(y,pred,th)
    specificity = TN/(TN+FP)
    return specificity



def get_ppv(y, pred, th=0.5):

    """
    Function to calculate the Postitive Predictive Value (PPV).
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    PPV = 0.0
    TP = get_true_pos(y,pred,th)
    FP = get_false_pos(y,pred,th)
    PPV = TP/(TP+FP) 
    return PPV



def get_npv(y, pred, th=0.5):

    """
    Function to calculate the Negative Predictive Value (NPV).
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    NPV = 0.0  
    TN = get_true_neg(y,pred,th)
    FN = get_false_neg(y,pred,th)
    NPV = TN/(TN+FN)    
    return NPV



def plot_graphs(history, metric):

    """
    Function to calculate the plot the evolution of the metrics during training.
    
    Inputs
    history: History object result of training a model with model.fit()
    metric: Name of the metric of interest (Accuracy, Loss, ...)
    """

    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()



def get_prevalence(y):
    
    """
    Function to compute the positive class prevalence (binary classification)

    Inputs
    y: List of labels
    """
    
    prevalence = 0.0
    prevalence = (1/len(y))*np.sum(y)
    return prevalence



def get_performance_metrics(y, pred, class_labels, tp=get_true_pos, tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg, acc=get_accuracy, prevalence=get_prevalence, sens=get_sensitivity, 
                            spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score,
                            thresholds=[]):
    
    """
    Function to compute a variety of performance metrics for a classification model.

    Inputs
    y: Ground truth labels
    pred: Predicted labels
    class_labels: List of class names
    """
    
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)
    columns = ["", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i][0] = class_labels[i]
        df.loc[i][1] = round(tp(y[:, i], pred[:, i]),
                             3) if tp != None else "Not Defined"
        df.loc[i][2] = round(tn(y[:, i], pred[:, i]),
                             3) if tn != None else "Not Defined"
        df.loc[i][3] = round(fp(y[:, i], pred[:, i]),
                             3) if fp != None else "Not Defined"
        df.loc[i][4] = round(fn(y[:, i], pred[:, i]),
                             3) if fn != None else "Not Defined"
        df.loc[i][5] = round(acc(y[:, i], pred[:, i], thresholds[i]),
                             3) if acc != None else "Not Defined"
        df.loc[i][6] = round(prevalence(y[:, i]),
                             3) if prevalence != None else "Not Defined"
        df.loc[i][7] = round(sens(y[:, i], pred[:, i], thresholds[i]),
                             3) if sens != None else "Not Defined"
        df.loc[i][8] = round(spec(y[:, i], pred[:, i], thresholds[i]),
                             3) if spec != None else "Not Defined"
        df.loc[i][9] = round(ppv(y[:, i], pred[:, i], thresholds[i]),
                             3) if ppv != None else "Not Defined"
        df.loc[i][10] = round(npv(y[:, i], pred[:, i], thresholds[i]),
                              3) if npv != None else "Not Defined"
        df.loc[i][11] = round(auc(y[:, i], pred[:, i]),
                              3) if auc != None else "Not Defined"
        df.loc[i][12] = round(f1(y[:, i], pred[:, i] > thresholds[i]),
                              3) if f1 != None else "Not Defined"
        df.loc[i][13] = round(thresholds[i], 3)
    df = df.set_index("")
    return df

# Metrics for semantic segmentation

def dice_coef(y_true, y_pred):

    """
    Function to calculate the DICE coefficient for 2 classes.
    
    Inputs
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask
    """

    smooth = 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def dice_coef_loss(y_true, y_pred):

    """
    Cost function based on the DICE coefficient to train segmentation models.
    
    Inputs
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask
    """

    return 1-dice_coef(y_true, y_pred)



def iou(y_true, y_pred):

    """
    Function to calculate the Intersection over Union (IoU).
    
    Inputs
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask
    """

    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        smooth = 1
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)



def bce_dice_loss(y_true, y_pred):

    """
    Cost function based on the DICE coefficient and Binary Crossentropy loss to train segmentation models.
    
    Inputs
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask
    """

    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)



def focal_loss(y_true, y_pred):
    
    """
    Implementation of the Focal Loss cost function to train segmentation models.
    
    Inputs
    y_true: Ground truth segmentation mask
    y_pred: Predicted segmentation mask
    """

    alpha=0.25
    gamma=2
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)