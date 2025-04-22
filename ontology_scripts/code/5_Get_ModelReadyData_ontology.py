#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:22:20 2021

@author: lucasliu
"""

import pandas as pd
import numpy as np
import os
import h5py
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler   


def Load_ontologydata(data_dir):
    #Read file
    ent_emb = pd.read_csv(data_dir + "ent_embeddings.csv", index_col=0)
    target_emb = ent_emb.iloc[0,:]
    other_emb =  ent_emb.iloc[1:,:]
    
    target_emb = target_emb.to_numpy()
    target_emb = target_emb.reshape(target_emb.shape[0],1)
   

    other_emb = other_emb.to_numpy()
    other_emb = np.transpose(other_emb) #rows: embedding index, col: ontology (high_SBP,LowSBP,lowScr,Ab_RR)


    
    rel_emb = pd.read_csv(data_dir + "rel_embeddings.csv", index_col=0)
    rel_emb = rel_emb.iloc[:,1:] #drop fisrt n_ca col
    rel_emb = rel_emb.to_numpy() 
    rel_emb = np.transpose(rel_emb) #rows: embedding index, col: ontology (#AKItohighSBP, AKItolowSBP,AKItoScr,AKItoAbRR)
 
    
    #scale/normalize to center
    scalers = c()    
    target_emb = scalers.fit_transform(target_emb) 
    scalers = MinMaxScaler()    
    other_emb = scalers.fit_transform(other_emb) 
    scalers = MinMaxScaler()    
    rel_emb = scalers.fit_transform(rel_emb) 

    target_emb =  torch.FloatTensor(target_emb)
    other_emb =  torch.FloatTensor(other_emb)
    rel_emb =  torch.FloatTensor(rel_emb)
    
    return target_emb,other_emb,rel_emb


#Input dir
proj_dir  = "./"
data_dir1 = "./Ontology_features/"
outdir = "./Ontology_features/"

#ontology embeddings
target_emb,other_emb,rel_emb = Load_ontologydata(data_dir1)


#Output ontology embeddings
hf_onto = h5py.File(outdir + 'Ontology_Embeddings.h5', 'w')
hf_onto.create_dataset('target_emb', data=target_emb)
hf_onto.create_dataset('other_emb',  data=other_emb)
hf_onto.create_dataset('rel_emb',    data=rel_emb)
hf_onto.close()