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


class MyLSTM(Model):
  def __init__(self, timesteps, n_feature,n_feature2,latent_dim):
    super(MyLSTM, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    


    self.Masking1 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature))
    self.LSTM1 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2))
    self.LSTM2 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)


    self.d1 = Dense(1, activation='tanh')
    self.flatten = Flatten()
    self.act = Activation('softmax')
    self.rv = RepeatVector(latent_dim*2) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    self.Dropout = Dropout(0.2)
    self.d2 = Dense(1, activation='sigmoid')
  def call(self, inputs1,inputs2,training=None):
    x1 = self.Masking1(inputs1)
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(inputs2)
    x2 = self.LSTM2(x2)  
    
    merged = concatenate([x1,x2],axis=2)
    
    #Attention
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention = Permute([2, 1])(attention)


    sent_representation1 = Multiply()([merged, attention]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*2,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    #sent_representation = self.Dropout(sent_representation,training=training)
    #sent_representation = self.Dropout(sent_representation)

    y = self.d2(sent_representation)



    return y

class MyLSTM_4grps(Model):
  def __init__(self, timesteps, n_feature,n_feature2,n_feature3,n_feature4,latent_dim):
    super(MyLSTM_4grps, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    self.n_feature3 = n_feature3 
    self.n_feature4 = n_feature4
    


    self.Masking1 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature))
    self.LSTM1 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2))
    self.LSTM2 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking3 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature3))
    self.LSTM3 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking4 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature4))
    self.LSTM4 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)

    self.d1 = Dense(1, activation='tanh')
    self.flatten = Flatten()
    self.act = Activation('softmax')
    self.rv = RepeatVector(latent_dim*4) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    self.Dropout = Dropout(0.2)
    self.d2 = Dense(1, activation='sigmoid')
  def call(self, inputs1,inputs2,inputs3,inputs4,training=None):
    x1 = self.Masking1(inputs1)
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(inputs2)
    x2 = self.LSTM2(x2)  
    
    x3 = self.Masking3(inputs3)
    x3 = self.LSTM3(x3) 
    
    x4 = self.Masking4(inputs4)
    x4 = self.LSTM4(x4) 
    
    merged = concatenate([x1,x2,x3,x4],axis=2)
    
    #Attention
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention = Permute([2, 1])(attention)


    sent_representation1 = Multiply()([merged, attention]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*4,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    #sent_representation = self.Dropout(sent_representation,training=training)
    #sent_representation = self.Dropout(sent_representation)
    y = self.d2(sent_representation)



    return y

#################################################################################################
#The following model has the same architecture as the above ones, 
#But they take one input argument so that the fit() function can be used for non-customize training
#################################################################################################

#This function takes one input argument
class MyLSTM2_2grps(Model):
  def __init__(self, timesteps, n_feature,n_feature2,latent_dim):
    super(MyLSTM2_2grps, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    


    self.Masking1 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature))
    self.LSTM1 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2))
    self.LSTM2 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)


    self.d1 = Dense(1, activation='tanh')
    self.flatten = Flatten()
    self.act = Activation('softmax')
    self.rv = RepeatVector(latent_dim*2) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    self.Dropout = Dropout(0.2)
    self.d2 = Dense(1, activation='sigmoid')
  def call(self, inputs):
    x1 = self.Masking1(inputs[0])
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(inputs[1])
    x2 = self.LSTM2(x2)  
    
    merged = concatenate([x1,x2],axis=2)
    
    #Attention
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention = Permute([2, 1])(attention)


    sent_representation1 = Multiply()([merged, attention]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*2,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    #sent_representation = self.Dropout(sent_representation)
    y = self.d2(sent_representation)



    return y

#This function takes one input argument
class MyLSTM2_4grps(Model):
  def __init__(self, timesteps, n_feature,n_feature2,n_feature3,n_feature4,latent_dim):
    super(MyLSTM2_4grps, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    self.n_feature3 = n_feature3 
    self.n_feature4 = n_feature4
    


    self.Masking1 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature))
    self.LSTM1 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2))
    self.LSTM2 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking3 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature3))
    self.LSTM3 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking4 = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature4))
    self.LSTM4 = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)

    self.d1 = Dense(1, activation='tanh')
    self.flatten = Flatten()
    self.act = Activation('softmax')
    self.rv = RepeatVector(latent_dim*4) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    self.Dropout = Dropout(0.2)
    self.d2 = Dense(1, activation='sigmoid')
  def call(self, inputs):
    x1 = self.Masking1(inputs[0])
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(inputs[1])
    x2 = self.LSTM2(x2)  
    
    x3 = self.Masking3(inputs[2])
    x3 = self.LSTM3(x3) 
    
    x4 = self.Masking4(inputs[3])
    x4 = self.LSTM4(x4) 
    
    merged = concatenate([x1,x2,x3,x4],axis=2)
    
    #Attention
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention = Permute([2, 1])(attention)


    sent_representation1 = Multiply()([merged, attention]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*4,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    #sent_representation = self.Dropout(sent_representation)
    y = self.d2(sent_representation)



    return y
