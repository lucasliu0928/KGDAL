#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:33:47 2021

@author: lucasliu
"""
import numpy as np

def get_resampled_idxes(n_to_resample,n_to_match):
    candidates_idxes = np.arange(n_to_resample)
    choices_idxes = np.random.choice(candidates_idxes, n_to_match) #select candidates_idxes to match n_to_match
    return choices_idxes

def get_resampled_concat_feature_label(resampled_idxes,feature_list_to_resample,feature_list_notto_resample,order):
        res_feature_list = []
        for feature_df in feature_list_to_resample:
            res_feature_list.append(feature_df[resampled_idxes])
        
        #concatenate resampled feature to non resampled
        concated_feature_list=[]
        for i in range(len(res_feature_list)):
            #concatenate
            conct_df = np.concatenate([res_feature_list[i], feature_list_notto_resample[i]], axis=0)
            #Shuffling
            conct_df = conct_df[order]
            concated_feature_list.append(conct_df)
        return concated_feature_list
    
def oversampling_func(X1,X2,y):
    
    #Convert to numpy array
    X1 = X1.numpy()
    X2 = X2.numpy()
    y = y.numpy()
    
    ##Get pos and negtive indexes
    pos_idxes = np.where(y==1)[0]
    neg_idxes = np.where(y==0)[0]
    n_pos = len(pos_idxes)
    n_neg = len(neg_idxes)
    
    #Get pos and neg feature
    pos_featuresA = X1[pos_idxes]
    pos_featuresB = X2[pos_idxes]
    
    neg_featuresA = X1[neg_idxes]
    neg_featuresB = X2[neg_idxes]
    
    #Get pos and neg label
    pos_labels = y[pos_idxes]
    neg_labels = y[neg_idxes]
    
    
    if (n_pos < n_neg): #If n of positve is < n of neg
        np.random.seed(0)
        resampled_idxes = get_resampled_idxes(n_pos,n_neg) #select pos index to match the n of neg

        #Get selected features and labels
        feature_list_to_resample = [pos_featuresA,pos_featuresB,pos_labels] 
        feature_list_notto_resample = [neg_featuresA,neg_featuresB,neg_labels]
        order = np.arange(len(resampled_idxes)*2) #shuffling order 
        np.random.shuffle(order) #Shuffling
        X1,X2,y = get_resampled_concat_feature_label(resampled_idxes,feature_list_to_resample,feature_list_notto_resample,order)

        
    elif(n_pos > n_neg):
        np.random.seed(0)
        resampled_idxes = get_resampled_idxes(n_neg,n_pos) #select neg index to match the n of pos

        #Get selected features and labels
        feature_list_to_resample = [neg_featuresA,neg_featuresB,neg_labels]
        feature_list_notto_resample = [pos_featuresA,pos_featuresB,pos_labels] 
        order = np.arange(len(resampled_idxes)*2) #shuffling order 
        np.random.shuffle(order) #Shuffling
        X1,X2,y = get_resampled_concat_feature_label(resampled_idxes,feature_list_to_resample,feature_list_notto_resample,order)

    return X1,X2,y