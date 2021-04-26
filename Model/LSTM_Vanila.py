#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:55:27 2021

@author: lucasliu
"""

from tensorflow.keras.layers import Dense, Flatten, LSTM, Masking,RepeatVector,concatenate,Activation,Permute,Multiply,Lambda,Dropout,Add
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from keras.constraints import unit_norm,max_norm
from keras import backend as K


class VanillaLSTM(Model):
  def __init__(self, timesteps, n_feature,latent_dim):
    super(VanillaLSTM, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 


    self.Masking = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature))
    #self.batch_norm = BatchNormalization()
    self.LSTM = LSTM(latent_dim,activation='relu', 
        kernel_constraint = unit_norm(),
        recurrent_constraint = max_norm(3),
        bias_constraint = max_norm(3),
        activity_regularizer=regularizers.l2(1e-2),
        kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
        bias_regularizer=regularizers.l2(1e-2))
    self.Dropout = Dropout(0.2)
    self.d = Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.Masking(inputs)
    #x = self.batch_norm(x)
    x = self.LSTM(x)   
    #x = self.Dropout(x)   
    
    y = self.d(x)

    return y
