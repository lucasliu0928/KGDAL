#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:07:00 2021

@author: lucasliu
"""

import numpy as np
import pandas as pd
import math
import sys
import os
import tensorflow as tf

def load_data8(data_dir,static_f_flag,feature_group):
    dbp_df = pd.read_csv(data_dir + "DBP_df.csv",index_col= 0)  #_TIDs.csv
    sbp_df = pd.read_csv(data_dir + "SBP_df.csv",index_col= 0) 
    cre_df = pd.read_csv(data_dir + "SCR_df.csv",index_col= 0) 
    Bicarbonate_df = pd.read_csv(data_dir + "Bicarbonate_df.csv",index_col= 0) 
    Hematocrit_df = pd.read_csv(data_dir + "Hematocrit_df.csv",index_col= 0) 
    Potassium_df = pd.read_csv(data_dir + "Potassium_df.csv",index_col= 0) 
    
    Billirubin_df = pd.read_csv(data_dir + "Billirubin_df.csv",index_col= 0) 
    FIO2_df = pd.read_csv(data_dir + "FIO2_df.csv",index_col= 0) 
    PH_df = pd.read_csv(data_dir + "PH_df.csv",index_col= 0) 
    PO2_df = pd.read_csv(data_dir + "PO2_df.csv",index_col= 0) 
    Sodium_df = pd.read_csv(data_dir + "Sodium_df.csv",index_col= 0) 
    Temp_df = pd.read_csv(data_dir + "Temp_df.csv",index_col= 0) 
    WBC_df = pd.read_csv(data_dir + "WBC_df.csv",index_col= 0) 
    HR_df = pd.read_csv(data_dir + "HR_df.csv",index_col= 0) 
    RR_df = pd.read_csv(data_dir + "RR_df.csv",index_col= 0) 
    inICU_df = pd.read_csv(data_dir + "inICU_df.csv",index_col= 0) 
    onCRRT_df = pd.read_csv(data_dir + "onCRRT_df.csv",index_col= 0) 
    onHD_df = pd.read_csv(data_dir + "onHD_df.csv",index_col= 0) 
    onRRT_df = pd.read_csv(data_dir + "onRRT_df.csv",index_col= 0)  
    onMV_df = pd.read_csv(data_dir + "onMV_df.csv",index_col= 0)  
    
    records_th = dbp_df.shape[1] #hours
    
    #load overlapping time info
    overlap_df = pd.read_csv(data_dir + "Final_overlap_df.csv",index_col= 'ENCNTR_ID')  
    overlap_df = overlap_df.drop(["Unnamed: 0",'Start_time', 'End_time'], axis=1)
 
    #outcome 
    outcome_df = pd.read_csv(data_dir + "Final_Outcome_df.csv",index_col= None) 
    outcome_df = outcome_df.drop("Unnamed: 0", axis=1)
    outcome_df = outcome_df.sample(frac = 1,random_state = 1)  #just shuffling
    outcome_df = outcome_df.reset_index(drop=True)
  
    #reorder dfs by outcome ID order
    dbp_df = dbp_df.reindex(outcome_df['ENCNTR_ID'])
    sbp_df = sbp_df.reindex(outcome_df['ENCNTR_ID'])
    cre_df = cre_df.reindex(outcome_df['ENCNTR_ID'])
    Bicarbonate_df = Bicarbonate_df.reindex(outcome_df['ENCNTR_ID'])
    Hematocrit_df = Hematocrit_df.reindex(outcome_df['ENCNTR_ID'])
    Potassium_df = Potassium_df.reindex(outcome_df['ENCNTR_ID'])
    
    Billirubin_df = Billirubin_df.reindex(outcome_df['ENCNTR_ID'])
    FIO2_df = FIO2_df.reindex(outcome_df['ENCNTR_ID'])
    PH_df = PH_df.reindex(outcome_df['ENCNTR_ID'])
    PO2_df = PO2_df.reindex(outcome_df['ENCNTR_ID'])
    Sodium_df = Sodium_df.reindex(outcome_df['ENCNTR_ID'])
    Temp_df = Temp_df.reindex(outcome_df['ENCNTR_ID'])
    WBC_df = WBC_df.reindex(outcome_df['ENCNTR_ID'])
    HR_df = HR_df.reindex(outcome_df['ENCNTR_ID'])
    RR_df = RR_df.reindex(outcome_df['ENCNTR_ID'])
    inICU_df = inICU_df.reindex(outcome_df['ENCNTR_ID'])
    onCRRT_df = onCRRT_df.reindex(outcome_df['ENCNTR_ID'])
    onHD_df = onHD_df.reindex(outcome_df['ENCNTR_ID'])
    onRRT_df = onRRT_df.reindex(outcome_df['ENCNTR_ID'])
    onMV_df = onMV_df.reindex(outcome_df['ENCNTR_ID'])
    
    overlap_df = overlap_df.reindex(outcome_df['ENCNTR_ID'])
    
    ##Train and test index
    train_ID_df = pd.read_csv(data_dir + "train_ids.csv",index_col= 0) 
    test_ID_df = pd.read_csv(data_dir + "test_ids.csv",index_col= 0) 
    external_validation_ID_df = pd.read_csv(data_dir + "external_test_ids.csv",index_col= 0) 
    
    train_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(train_ID_df['ENCNTR_ID'])] #627
    test_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(test_ID_df['ENCNTR_ID'])] #159
    external_Valid_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(external_validation_ID_df['ENCNTR_ID'])] #159


    #add all stationary features except the last one, the last one is outomce
    if (feature_group == "ALL") :
        dynamic_lists = [sbp_df,dbp_df,cre_df,Bicarbonate_df,Hematocrit_df,Potassium_df,
                             Billirubin_df,PH_df,PO2_df,Sodium_df,Temp_df,
                             WBC_df,HR_df,RR_df,onMV_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Cardiovascular"):
         dynamic_lists = [sbp_df,dbp_df,HR_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
         n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Metabolism"):
         dynamic_lists = [cre_df,Potassium_df,Sodium_df,Billirubin_df,Bicarbonate_df,Temp_df,PH_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
         n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Blood"):
         dynamic_lists = [Hematocrit_df,WBC_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
         n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Respiratory"):
         dynamic_lists = [RR_df,PO2_df,onMV_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
         n_of_dynamic_features = len(dynamic_lists)
         
    all_feature_lists = dynamic_lists
    
    if(static_f_flag == True):
        #TAKI feature
        Demo_df = pd.read_csv(data_dir + "final_demo_df.csv",index_col= 1) 
        Demo_df = Demo_df.drop("Unnamed: 0", axis=1)
        Demo_df = Demo_df.reindex(outcome_df['ENCNTR_ID'])
        Demo_df = Demo_df.fillna(Demo_df.mean()) #Weight and BMi has messing values, replace with mean

        #rep stationaory feature at every time step
        staiontary_feature_list=[]
        # Iterate over given columns only from the dataframe
        for column in Demo_df:
            # Select column contents by column name using [] operator
            columnSeriesObj = Demo_df[column].to_frame()
            sta_feature_df = pd.concat([columnSeriesObj]*records_th, axis=1)
            staiontary_feature_list.append(sta_feature_df)

        for station_f in staiontary_feature_list: 
            all_feature_lists.append(station_f)
        
    all_feature_df = np.stack(all_feature_lists,axis=2) #[Patients,Time,Feature]

    
    return all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features

#exclude the extra feature: PO2,PH and MV
def load_data9(data_dir,static_f_flag,feature_group, data_duration_type):
    dbp_df = pd.read_csv(data_dir + "DBP_df.csv",index_col= 0)  #_TIDs.csv
    sbp_df = pd.read_csv(data_dir + "SBP_df.csv",index_col= 0) 
    cre_df = pd.read_csv(data_dir + "SCR_df.csv",index_col= 0) 
    Bicarbonate_df = pd.read_csv(data_dir + "Bicarbonate_df.csv",index_col= 0) 
    Hematocrit_df = pd.read_csv(data_dir + "Hematocrit_df.csv",index_col= 0) 
    Potassium_df = pd.read_csv(data_dir + "Potassium_df.csv",index_col= 0) 
    
    Billirubin_df = pd.read_csv(data_dir + "Billirubin_df.csv",index_col= 0) 
    Sodium_df = pd.read_csv(data_dir + "Sodium_df.csv",index_col= 0) 
    Temp_df = pd.read_csv(data_dir + "Temp_df.csv",index_col= 0) 
    WBC_df = pd.read_csv(data_dir + "WBC_df.csv",index_col= 0) 
    HR_df = pd.read_csv(data_dir + "HR_df.csv",index_col= 0) 
    RR_df = pd.read_csv(data_dir + "RR_df.csv",index_col= 0) 
    inICU_df = pd.read_csv(data_dir + "inICU_df.csv",index_col= 0) 
    onCRRT_df = pd.read_csv(data_dir + "onCRRT_df.csv",index_col= 0) 
    onHD_df = pd.read_csv(data_dir + "onHD_df.csv",index_col= 0) 
    onRRT_df = pd.read_csv(data_dir + "onRRT_df.csv",index_col= 0)  
    
    records_th = dbp_df.shape[1] #hours
    
    #load overlapping time info
    overlap_df = pd.read_csv(data_dir + "Final_overlap_df.csv",index_col= 'ENCNTR_ID')  
    overlap_df = overlap_df.drop(["Unnamed: 0",'Start_time', 'End_time'], axis=1)
 
    #outcome 
    outcome_df = pd.read_csv(data_dir + "Final_Outcome_df.csv",index_col= None) 
    outcome_df = outcome_df.drop("Unnamed: 0", axis=1)
    outcome_df = outcome_df.sample(frac = 1,random_state = 1)  #just shuffling
    outcome_df = outcome_df.reset_index(drop=True)
  
    #reorder dfs by outcome ID order
    dbp_df = dbp_df.reindex(outcome_df['ENCNTR_ID'])
    sbp_df = sbp_df.reindex(outcome_df['ENCNTR_ID'])
    cre_df = cre_df.reindex(outcome_df['ENCNTR_ID'])
    Bicarbonate_df = Bicarbonate_df.reindex(outcome_df['ENCNTR_ID'])
    Hematocrit_df = Hematocrit_df.reindex(outcome_df['ENCNTR_ID'])
    Potassium_df = Potassium_df.reindex(outcome_df['ENCNTR_ID'])
    
    Billirubin_df = Billirubin_df.reindex(outcome_df['ENCNTR_ID'])
    Sodium_df = Sodium_df.reindex(outcome_df['ENCNTR_ID'])
    Temp_df = Temp_df.reindex(outcome_df['ENCNTR_ID'])
    WBC_df = WBC_df.reindex(outcome_df['ENCNTR_ID'])
    HR_df = HR_df.reindex(outcome_df['ENCNTR_ID'])
    RR_df = RR_df.reindex(outcome_df['ENCNTR_ID'])
    inICU_df = inICU_df.reindex(outcome_df['ENCNTR_ID'])
    onCRRT_df = onCRRT_df.reindex(outcome_df['ENCNTR_ID'])
    onHD_df = onHD_df.reindex(outcome_df['ENCNTR_ID'])
    onRRT_df = onRRT_df.reindex(outcome_df['ENCNTR_ID'])
    
    overlap_df = overlap_df.reindex(outcome_df['ENCNTR_ID'])
    
    ##Train and test index
    train_ID_df = pd.read_csv(data_dir + "train_ids.csv",index_col= 0) 
    test_ID_df = pd.read_csv(data_dir + "test_ids.csv",index_col= 0) 
    external_validation_ID_df = pd.read_csv(data_dir + "external_test_ids.csv",index_col= 0) 
    
    train_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(train_ID_df['ENCNTR_ID'])] #627
    test_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(test_ID_df['ENCNTR_ID'])] #159
    external_Valid_idx = outcome_df.index[outcome_df['ENCNTR_ID'].isin(external_validation_ID_df['ENCNTR_ID'])] #159


    #add all stationary features except the last one, the last one is outomce
    if (feature_group == "ALL") :
        if (data_duration_type == "inHOSP"): #if in HOSP
            dynamic_lists = [sbp_df,dbp_df,cre_df,Bicarbonate_df,Hematocrit_df,Potassium_df,
                                 Billirubin_df,Sodium_df,Temp_df,
                                 WBC_df,HR_df,RR_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "inICU"): #if in ICU data, exclude in ICU flag
            dynamic_lists = [sbp_df,dbp_df,cre_df,Bicarbonate_df,Hematocrit_df,Potassium_df,
                                 Billirubin_df,Sodium_df,Temp_df,
                                 WBC_df,HR_df,RR_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "onRRT"): #if on RRT data,  exclude on RRT flag
            dynamic_lists = [sbp_df,dbp_df,cre_df,Bicarbonate_df,Hematocrit_df,Potassium_df,
                                 Billirubin_df,Sodium_df,Temp_df,
                                 WBC_df,HR_df,RR_df,inICU_df,onCRRT_df,onHD_df]
        n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Cardiovascular"):
        if (data_duration_type == "inHOSP"):
            dynamic_lists = [sbp_df,dbp_df,HR_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "inICU"): #if in ICU data, exclude in ICU flag
            dynamic_lists = [sbp_df,dbp_df,HR_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "onRRT"): #if on RRT data,  exclude on RRT flag
            dynamic_lists = [sbp_df,dbp_df,HR_df,inICU_df,onCRRT_df,onHD_df]
        n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Metabolism"):
        if (data_duration_type == "inHOSP"):
            dynamic_lists = [cre_df,Potassium_df,Sodium_df,Billirubin_df,Bicarbonate_df,Temp_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "inICU"): 
            dynamic_lists = [cre_df,Potassium_df,Sodium_df,Billirubin_df,Bicarbonate_df,Temp_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "onRRT"):
            dynamic_lists = [cre_df,Potassium_df,Sodium_df,Billirubin_df,Bicarbonate_df,Temp_df,inICU_df,onCRRT_df,onHD_df]
        n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Blood"):
        if (data_duration_type == "inHOSP"):
            dynamic_lists = [Hematocrit_df,WBC_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "inICU"):
            dynamic_lists = [Hematocrit_df,WBC_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "onRRT"): #if on RRT data,  Keep  inICU onCRRT, onHD, do not keep onRRT flag
            dynamic_lists = [Hematocrit_df,WBC_df,inICU_df,onCRRT_df,onHD_df]
        n_of_dynamic_features = len(dynamic_lists)
    elif (feature_group == "Respiratory"):
        if (data_duration_type == "inHOSP"):
            dynamic_lists = [RR_df,inICU_df,onCRRT_df,onHD_df,onRRT_df]
        elif (data_duration_type == "inICU"): 
            dynamic_lists = [RR_df,onCRRT_df,onHD_df,onRRT_df]       
        elif (data_duration_type == "onRRT"): #if on RRT data,  Keep  inICU onCRRT, onHD, do not keep onRRT flag
            dynamic_lists = [RR_df,inICU_df,onCRRT_df,onHD_df]
        n_of_dynamic_features = len(dynamic_lists)
         
    all_feature_lists = dynamic_lists
    
    if(static_f_flag == True):
        #TAKI feature
        Demo_df = pd.read_csv(data_dir + "final_demo_df.csv",index_col= 1) 
        Demo_df = Demo_df.drop("Unnamed: 0", axis=1)
        Demo_df = Demo_df.reindex(outcome_df['ENCNTR_ID'])
        Demo_df = Demo_df.fillna(Demo_df.mean()) #Weight and BMi has messing values, replace with mean

        #rep stationaory feature at every time step
        staiontary_feature_list=[]
        # Iterate over given columns only from the dataframe
        for column in Demo_df:
            # Select column contents by column name using [] operator
            columnSeriesObj = Demo_df[column].to_frame()
            sta_feature_df = pd.concat([columnSeriesObj]*records_th, axis=1)
            staiontary_feature_list.append(sta_feature_df)

        for station_f in staiontary_feature_list: 
            all_feature_lists.append(station_f)
        
    all_feature_df = np.stack(all_feature_lists,axis=2) #[Patients,Time,Feature]

    
    return all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features

def minmax_norm_func(in_data):
    #This fucntion min-max normlization over feature space
    #for each feature data(row:pt, col: time steps), apply normalization for  each non-nan entry
    n_of_features = in_data.shape[2]
    for f in range(n_of_features):
        curr =  in_data[:, :, f]
        max_of_curr_feature = np.nanmax(curr)
        min_of_curr_feature = np.nanmin(curr)
        for i in range(curr.shape[0]):
              for j in range(curr.shape[1]):
                if (math.isnan(curr[i,j]) == False):
                      curr[i,j] = (curr[i,j]-min_of_curr_feature)/(max_of_curr_feature - min_of_curr_feature)
        in_data[:, :, f] = curr #store scaled value
    return in_data

def masking_func(in_data,n_of_dynamic_features,masking_val):
    n_of_patient = in_data.shape[0]
    #this is actual masking in masking layer in the model, when every feature is -1 at the time step
    #for each patients, mask out at time step of all features when all dynamic feature is nan
    for pts in range(n_of_patient):
        curr_pts_feature = in_data[pts,:,0:n_of_dynamic_features] #dynamic feature columns
        allnan_timestep = np.all(np.isnan(curr_pts_feature),axis = 1)
        in_data[pts,allnan_timestep,] = masking_val
    return in_data

#Get feature and outcome
def get_train_df(all_feature_df,outcome_df,overlap_df,outcome_col_name,n_of_dynamic_features,train_idx,test_idx,external_valid_idx):
    X = all_feature_df
    y = outcome_df[outcome_col_name]
    overlap_data = overlap_df

    X_train = X[train_idx,]
    X_test = X[test_idx,]
    X_Validation = X[external_valid_idx,]
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    y_Validation = y[external_valid_idx]
    
    #overlap data is also used for keep track of the IDs
    overlap_data_train = overlap_data.iloc[train_idx,]
    overlap_data_test = overlap_data.iloc[test_idx,]
    overlap_data_Validation = overlap_data.iloc[external_valid_idx,]
    
    
    #for each feature scale on the entire feature
    X_train = minmax_norm_func(X_train)
    X_test = minmax_norm_func(X_test)
    X_Validation = minmax_norm_func(X_Validation)
        
    #masking using -1 (acutal masking step)
    X_train = masking_func(X_train,n_of_dynamic_features,-1)
    X_test = masking_func(X_test,n_of_dynamic_features,-1)
    X_Validation = masking_func(X_Validation,n_of_dynamic_features,-1)
    
    #this masking is for treating missing feature at time step as -2, when other features are present
    #ex. Scr between every 6 hour
    X_train = np.nan_to_num(X_train,nan=-2) #fill Na with -1
    X_test = np.nan_to_num(X_test,nan=-2) #fill Na with -1
    X_Validation = np.nan_to_num(X_Validation,nan=-2) #fill Na with -1


    #convert label to numpy
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_Validation = y_Validation.to_numpy()

    #Count labels
    train_count = np.bincount(y_train)
    test_count = np.bincount(y_test)
    Validation_count = np.bincount(y_Validation)

    print("Train:","\nN Negtives:",train_count[0],"\nN Postives:",train_count[1])
    print("Test:","\nN Negtives:", test_count[0],"\nN Postives:",test_count[1])
    print("External Validation:","\nN Negtives:",Validation_count[0],"\nN Postives:",Validation_count[1])


    return X_train,X_test,X_Validation,y_train,y_test,y_Validation,overlap_data_train,overlap_data_test,overlap_data_Validation

def main_load_data_4grps(data_dir,outcome_col_name, data_duration_type):
    
    feature_grp1 = "Cardiovascular"
    feature_grp2 = "Metabolism"
    feature_grp3 = "Blood"
    feature_grp4 = "Respiratory"
    

    ##Load data A
    all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features = load_data9(data_dir,True,feature_grp1, data_duration_type)
    res = get_train_df(all_feature_df,outcome_df,overlap_df,outcome_col_name,n_of_dynamic_features,train_idx,test_idx,external_Valid_idx)
    X_train_A = res[0]
    X_test_A = res[1]
    X_Validation_A = res[2]
    y_train = res[3]
    y_test = res[4]
    y_Validation = res[5]
    train_overlap = res[6]
    test_overlap = res[7]
    validation_overlap = res[8]
    
    #Load data B
    all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features = load_data9(data_dir,True,feature_grp2, data_duration_type)
    res = get_train_df(all_feature_df,outcome_df,overlap_df,outcome_col_name,n_of_dynamic_features,train_idx,test_idx,external_Valid_idx)
    X_train_B = res[0]
    X_test_B = res[1]
    X_Validation_B = res[2]

    #Load data C
    all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features = load_data9(data_dir,True,feature_grp3, data_duration_type)
    res = get_train_df(all_feature_df,outcome_df,overlap_df,outcome_col_name,n_of_dynamic_features,train_idx,test_idx,external_Valid_idx)
    X_train_C = res[0]
    X_test_C = res[1]
    X_Validation_C = res[2]

    
    #Load data D
    all_feature_df,outcome_df,overlap_df,train_idx,test_idx,external_Valid_idx,n_of_dynamic_features = load_data9(data_dir,True,feature_grp4, data_duration_type)
    res = get_train_df(all_feature_df,outcome_df,overlap_df,outcome_col_name,n_of_dynamic_features,train_idx,test_idx,external_Valid_idx)
    X_train_D = res[0]
    X_test_D = res[1]
    X_Validation_D = res[2]

    #use this to check if we get the same order
    # y_train_B = res[3]
    # y_test_B = res[4]
    # y_Validation_B = res[5]
    # overlap_train_B = res[6]
    # overlap_test_B = res[7]
    # overlap_validation_B = res[8]
    
    #Data and label list
    train_datalist = [X_train_A,X_train_B,X_train_C,X_train_D]
    test_datalist = [X_test_A,X_test_B,X_test_C,X_test_D]
    validation_datalist = [X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D]

    return train_datalist,test_datalist,validation_datalist,y_train,y_test,y_Validation, train_overlap,test_overlap,validation_overlap,outcome_df,external_Valid_idx

