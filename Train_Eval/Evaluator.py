#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:37:06 2021

@author: lucasliu
"""

import tensorflow as tf
import pandas as pd

from Model.LSTM_ATTonTime import MyLSTM,MyLSTM_4grps
from Model.LSTM_ATTonTimeAndFeature_WithThresFeatures import AttnOnFeatures_ScaleAtt_4grps_withThresFeature
from Model.LSTM_Vanila import VanillaLSTM
from Ultility import Evaluation_funcs

#tf.keras.backend.set_floatx('float32')

def external_eval_1grp(ckpt_idx_to_restore,X_Validation,y_Validation,timesteps,n_feature,latent_dim,outdir):
    #restore model
    reconstructed_model = VanillaLSTM(timesteps,n_feature,latent_dim)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), mod = reconstructed_model)
    #manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint).expect_partial()

    ckpt.restore(outdir + '/tf_ckpts/ckpt-' + str(ckpt_idx_to_restore)).expect_partial() #use expect_partial to only restore vars used for validation, removed warnings
    
    pred_prob = reconstructed_model.predict(X_Validation, verbose=0)
    pred_classes = Evaluation_funcs.compute_performance2(y_Validation,pred_prob,False,0.5) #return performance at roc cutoff point
    accuracy,precision1,recall1,f11,precision0,recall0,f10 = Evaluation_funcs.compute_performance1(pred_classes,y_Validation) #return performance at threhold 0.5
    roc_auc = Evaluation_funcs.roc(y_Validation,pred_prob,False)
    pr_auc =  Evaluation_funcs.PR_AUC(y_Validation, pred_prob)
    F1_Class0,F3_Class1 = Evaluation_funcs.F_beta(y_Validation,pred_classes)
    
    perf_tb = pd.DataFrame([[accuracy,roc_auc,pr_auc,precision1,recall1,f11,precision0,recall0,f10,F1_Class0,F3_Class1]],columns=['ACC','ROC_AUC',"PR_AUC",'PREC1','RECALL1','F1_1','PREC0','RECALL0','F1_0','F1_Class0','F3_Class1'])
    perf_tb.to_csv(outdir + '/perf0314.csv')


#Two grps
def external_eval(ckpt_idx_to_restore,X_Validation_A,X_Validation_B,y_Validation,timesteps,n_featureA,n_featureB,outdir):
    #restore model
    reconstructed_model = MyLSTM(timesteps,n_featureA,n_featureB,8)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), mod = reconstructed_model)
    #manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint).expect_partial()

    ckpt.restore(outdir + '/tf_ckpts/ckpt-' + str(ckpt_idx_to_restore)).expect_partial() #use expect_partial to only restore vars used for validation, removed warnings
    pred_prob = reconstructed_model(X_Validation_A,X_Validation_B, training=False)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    t_loss2 = loss_object(y_Validation, pred_prob)
    
    pred_classes = Evaluation_funcs.compute_performance2(y_Validation,pred_prob,False,0.5) #return performance at roc cutoff point
    accuracy,precision1,recall1,f11,precision0,recall0,f10 = Evaluation_funcs.compute_performance1(pred_classes,y_Validation) #return performance at threhold 0.5
    
    roc_auc = Evaluation_funcs.roc(y_Validation,pred_prob,False)
    pr_auc =  Evaluation_funcs.PR_AUC(y_Validation, pred_prob)
    F1_Class0,F3_Class1 = Evaluation_funcs.F_beta(y_Validation,pred_classes)
    perf_tb = pd.DataFrame([[accuracy,roc_auc,pr_auc,precision1,recall1,f11,precision0,recall0,f10,F1_Class0,F3_Class1]],columns=['ACC','ROC_AUC',"PR_AUC",'PREC1','RECALL1','F1_1','PREC0','RECALL0','F1_0','F1_Class0','F3_Class1'])
    perf_tb.to_csv(outdir + '/perf0311.csv')
    
    return t_loss2

def external_eval_4grps(ckpt_idx_to_restore,X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D,y_Validation,timesteps,n_featureA,n_featureB,n_featureC,n_featureD,outdir):
    #restore model
    reconstructed_model = MyLSTM_4grps(timesteps,n_featureA,n_featureB,n_featureC,n_featureD,8)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), mod = reconstructed_model)
    #manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint).expect_partial()

    ckpt.restore(outdir + '/tf_ckpts/ckpt-' + str(ckpt_idx_to_restore)).expect_partial() #use expect_partial to only restore vars used for validation, removed warnings
    pred_prob = reconstructed_model(X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D, training=False)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    t_loss2 = loss_object(y_Validation, pred_prob)
    
    pred_classes = Evaluation_funcs.compute_performance2(y_Validation,pred_prob,False,0.5) #return performance at roc cutoff point
    accuracy,precision1,recall1,f11,precision0,recall0,f10 = Evaluation_funcs.compute_performance1(pred_classes,y_Validation) #return performance at threhold 0.5
    
    roc_auc = Evaluation_funcs.roc(y_Validation,pred_prob,False)
    pr_auc =  Evaluation_funcs.PR_AUC(y_Validation, pred_prob)
    F1_Class0,F3_Class1 = Evaluation_funcs.F_beta(y_Validation,pred_classes)
    perf_tb = pd.DataFrame([[accuracy,roc_auc,pr_auc,precision1,recall1,f11,precision0,recall0,f10,F1_Class0,F3_Class1]],columns=['ACC','ROC_AUC',"PR_AUC",'PREC1','RECALL1','F1_1','PREC0','RECALL0','F1_0','F1_Class0','F3_Class1'])
    perf_tb.to_csv(outdir + '/perf0311.csv')
    
    return t_loss2


def external_eval_4grps_withThFeatures(ckpt_idx_to_restore,X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D,X_Validation_A_th,X_Validation_B_th,X_Validation_C_th,X_Validation_D_th,y_Validation,timesteps,n_featureA,n_featureB,n_featureC,n_featureD,n_features_A_th,n_features_B_th,n_features_C_th,n_features_D_th,outdir):
    #restore model
    reconstructed_model = AttnOnFeatures_ScaleAtt_4grps_withThresFeature(timesteps,n_featureA,n_featureB,n_featureC,n_featureD,n_features_A_th,n_features_B_th,n_features_C_th,n_features_D_th,8)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), mod = reconstructed_model)
    #manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint).expect_partial()

    ckpt.restore(outdir + '/tf_ckpts/ckpt-' + str(ckpt_idx_to_restore)).expect_partial() #use expect_partial to only restore vars used for validation, removed warnings
    pred_prob = reconstructed_model(X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D,X_Validation_A_th,X_Validation_B_th,X_Validation_C_th,X_Validation_D_th, training=False)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    t_loss2 = loss_object(y_Validation, pred_prob)
    
    pred_classes = Evaluation_funcs.compute_performance2(y_Validation,pred_prob,False,0.5) #return performance at roc cutoff point
    accuracy,precision1,recall1,f11,precision0,recall0,f10 = Evaluation_funcs.compute_performance1(pred_classes,y_Validation) #return performance at threhold 0.5
    
    roc_auc = Evaluation_funcs.roc(y_Validation,pred_prob,False)
    pr_auc =  Evaluation_funcs.PR_AUC(y_Validation, pred_prob)
    F1_Class0,F3_Class1 = Evaluation_funcs.F_beta(y_Validation,pred_classes)
    perf_tb = pd.DataFrame([[accuracy,roc_auc,pr_auc,precision1,recall1,f11,precision0,recall0,f10,F1_Class0,F3_Class1]],columns=['ACC','ROC_AUC',"PR_AUC",'PREC1','RECALL1','F1_1','PREC0','RECALL0','F1_0','F1_Class0','F3_Class1'])
    perf_tb.to_csv(outdir + '/perf0315.csv')
    
    return t_loss2
