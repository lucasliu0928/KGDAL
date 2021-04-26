#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:11:37 2021

@author: lucasliu
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import fbeta_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_performance1(y_pred_classes , y_true, print_res):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    #Class 1:
    # precision tp / (tp + fp)
    precision1 = precision_score(y_true, y_pred_classes,pos_label=1)
    # recall: tp / (tp + fn)
    recall1 = recall_score(y_true, y_pred_classes,pos_label=1)
    # f1: 2 tp / (2 tp + fp + fn)
    f11 = f1_score(y_true, y_pred_classes,pos_label=1)
    F3_Class1 = fbeta_score(y_true, y_pred_classes, beta=3,pos_label= 1) #beta > 1, more on recall
 
    #Class 0:
    precision0 = precision_score(y_true, y_pred_classes,pos_label=0)
    recall0 = recall_score(y_true, y_pred_classes,pos_label=0)
    f10 = f1_score(y_true, y_pred_classes,pos_label=0)
    
    
    if print_res == True:
        print('Accuracy: %.2f' % accuracy)
        print('Precision1: %.2f' % precision1)
        print('Recall1: %.2f' % recall1)
        print('F1 score1: %.2f' % f11)
        print('F3_Class1 %0.2f' % F3_Class1)  
        print('Precision0 : %.2f' % precision0)
        print('Recall0 : %.2f' % recall0)
        print('F1 score0: %.2f' % f10)

    return accuracy,precision1,recall1,f11,precision0,recall0,f10,F3_Class1
    
    
def compute_pred_class(y_true, y_prob,cutoff_th_flag,defined_thres):
    ####################################
    # The optimal cut off would be where tpr is high and fpr is low
    # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
    ####################################
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    all_perf = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    #print(all_perf)
    
    if (cutoff_th_flag == True) : 
        thres = all_perf['thresholds'].values[0]
    else: 
        thres = defined_thres
  
    y_pred_classes = []
    for pred_p in y_prob:
        if pred_p >= thres: #all_perf['thresholds'].values[0]
            y_pred_classes.append(1)
        else:
            y_pred_classes.append(0)

    return y_pred_classes
    
def roc(y_true, y_prob,graph,print_res):
    """Print ROC Statistics and Graph"""
    roc_auc = roc_auc_score(y_true, y_prob)
    if graph:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (Area = %0.3f)'% roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specifity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        print('ROC Curve saved to roc.png')
        plt.savefig('roc.png')
    elif print_res == True:
        print('ROC-AUC %0.2f' % roc_auc)   

    return roc_auc

def PR_AUC(y_true, y_prob,print_res):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall,precision)
    if print_res == True:
        print('PR-AUC %0.2f' % pr_auc)   

    return pr_auc

def plot_training_history(history,plot_dir):
    # plot history for loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()    
    plt.savefig(plot_dir + '/LOSS_plot.png')
    plt.clf()
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(plot_dir + '/ACC_plot.png')


def plot_training_history2(n_epochs,train_acc_epoch,test_acc_epoch,train_loss_epoch,test_loss_epoch, plot_dir):
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, train_acc_epoch, 'b', label='Train',color="darkblue")
    plt.plot(epochs, test_acc_epoch, 'b', label='Test',color="darkorange")
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(plot_dir + '/ACC_plot.png')
    plt.clf()
    
    # summarize history for accuracy
    plt.plot(epochs, train_loss_epoch, 'b', label='Train',color="darkblue")
    plt.plot(epochs, test_loss_epoch, 'b', label='Test',color="darkorange")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(plot_dir + '/LOSS_plot.png')
    plt.clf()

def get_all_performce_table(pred_prob,y_true,print_res):
    pred_classes = compute_pred_class(y_true,pred_prob,False,0.5) #return performance at roc cutoff point
    accuracy,precision1,recall1,f11,precision0,recall0,f10,F3_Class1 = compute_performance1(pred_classes,y_true,print_res) #return performance at threhold 0.5
    roc_auc = roc(y_true,pred_prob,False,print_res)
    pr_auc =  PR_AUC(y_true, pred_prob,print_res)

    perf_tb = pd.DataFrame([[roc_auc,accuracy,precision1,recall1,F3_Class1,precision0,recall0,f10,pr_auc]],columns=
                            ['ROC_AUC','ACC','PREC1','RECALL1','F3_Class1','PREC0','RECALL0','F1_Class0','PR_AUC'])
    return perf_tb

def get_predicted_prob_table(pred_prob,y_true,ID_val):
    #Use overlap df to get ID, they are in the same order as y (in GetModelReadyData function)
    #Return predicted probabilty
    pred_prob_df = pd.DataFrame({'ENCNTR_ID': ID_val, 'Predicted_Y': pred_prob.numpy().reshape(-1),'True_y': y_true})
    return pred_prob_df
