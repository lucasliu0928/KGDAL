#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 04:19:26 2021

@author: lucasliu
"""
from tensorflow.keras.layers import Dense, Flatten, LSTM, Masking,RepeatVector,concatenate,Activation,Permute,Multiply,Lambda,Dropout,Add,Softmax
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from keras.constraints import unit_norm,max_norm
from keras import backend as K
import tensorflow as tf

class AttnOnFeatures_ScaleAtt_4grps_withThresFeature(Model):
  def __init__(self, timesteps, n_feature,n_feature2,n_feature3,n_feature4,n_feature_th,n_feature2_th,n_feature3_th,n_feature4_th,latent_dim):
    super(AttnOnFeatures_ScaleAtt_4grps_withThresFeature, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    self.n_feature3 = n_feature3 
    self.n_feature4 = n_feature4
    self.n_feature_th = n_feature_th
    self.n_feature2_th = n_feature2_th
    self.n_feature3_th = n_feature3_th
    self.n_feature4_th = n_feature4_th
    

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
    
    #For threshold featreus
    self.Masking1_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature_th))
    self.LSTM1_th = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2_th))
    self.LSTM2_th = LSTM(latent_dim,activation='relu', #tanh
                      kernel_constraint = unit_norm(),
                      recurrent_constraint = max_norm(3),
                      bias_constraint = max_norm(3),
                      activity_regularizer=regularizers.l2(1e-2),
                      kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                      bias_regularizer=regularizers.l2(1e-2),
                      return_sequences=True)
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking3_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature3_th))
    self.LSTM3_th = LSTM(latent_dim,activation='relu', #tanh
                      kernel_constraint = unit_norm(),
                      recurrent_constraint = max_norm(3),
                      bias_constraint = max_norm(3),
                      activity_regularizer=regularizers.l2(1e-2),
                      kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                      bias_regularizer=regularizers.l2(1e-2),
                      return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking4_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature4_th))
    self.LSTM4_th = LSTM(latent_dim,activation='relu', #tanh
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
    
    #Attention over time layers
    self.d3 = Dense(1, activation='tanh')
    self.flatten2 = Flatten()
    self.act2 = Activation('softmax')
    self.rv2 = RepeatVector(timesteps) #the size here should be equal to the time steps
    
    #softmax layer for attention norm
    self.norm = Softmax(axis= -1)

  def call(self, inputs):
    # #get HPO Sim score
    #    hpo_dist = {"Cardiovascular": 0.0440939,"Metabolism": 0.109443,"Blood": 0.817214,"Respiratory": 0.975282}
    #    TransE_dist = {"Cardiovascular": 0.02150,"Metabolism": 0.33634,"Blood": 0.68064,"Respiratory": 0.97682}

    x1_score = 0.02150
    x2_score = 0.33634
    x3_score = 0.68064
    x4_score = 0.97682
    
    x1 = self.Masking1(inputs[0])
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(inputs[1])
    x2 = self.LSTM2(x2)  
    
    x3 = self.Masking3(inputs[2])
    x3 = self.LSTM3(x3) 
    
    x4 = self.Masking4(inputs[3])
    x4 = self.LSTM4(x4) 
    
    merged = concatenate([x1,x2,x3,x4],axis=2)
    
    #  #Attention On Time
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...

    #Threshold features
    x1_th = self.Masking1_th(inputs[4])
    x1_th = self.LSTM1_th(x1_th)   
    
    x2_th = self.Masking2_th(inputs[5])
    x2_th = self.LSTM2_th(x2_th)  
    
    x3_th = self.Masking3_th(inputs[6])
    x3_th = self.LSTM3_th(x3_th) 
    
    x4_th = self.Masking4_th(inputs[7])
    x4_th = self.LSTM4_th(x4_th) 
    
    merged2 = concatenate([x1,x2,x3,x4],axis=2)
     
    #Attention on Features
    merged_transp = Permute([2, 1])(merged2)
    attention_f = self.d3(merged_transp) 
    attention_f = self.flatten2(attention_f) 
    attention_f = self.act2(attention_f)
    attention_f = self.rv2(attention_f) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention_f = Permute([2, 1])(attention_f)
    
    #Multiply ontology similarity 
    attention_f1 = attention_f[:,0:8,:]*x1_score
    attention_f2 = attention_f[:,8:16,:]*x2_score
    attention_f3 = attention_f[:,16:32,:]*x3_score
    attention_f4 = attention_f[:,32:48,:]*x4_score
    attention_f5 = concatenate([attention_f1,attention_f2,attention_f3,attention_f4],axis=1)
    

    # #Add or multiply or new rules(Scaled by another one) to combine the two attention
    #attention_comb1 = Multiply()([attention_f5, attention]) #multiply
    #attention_comb = Add()([attention_f5, attention]) #add 
    
    #New rule: time attention scaled by feature attention
    #Find the scale for each rows in feature attention matrix
    v1 = tf.reshape(attention_f5[:,0,:],[tf.shape(attention_f5[:,0,:])[0],1 , tf.shape(attention_f5[:,0,:])[1]]) #get the first row vector
    scale_matrix = Multiply()([attention_f5, 1/v1])  #every raw divided the first row(feature atten)
    attention_comb = Multiply()([attention, scale_matrix]) #scale time att
     
    #TODO: The norm does not work
    #normalize  comb matrix for each sample
    #attention_comb_reshaped = tf.reshape(attention_comb, [tf.shape(attention_comb)[0],tf.shape(attention_comb)[1]*tf.shape(attention_comb)[2]]) #reshape to [sample,all cells for this sample]
    #attention_comb_norm = self.norm(attention_comb_reshaped)
    #attention_comb_norm = tf.reshape(attention_comb_norm, [attention_comb.shape[0],attention_comb.shape[1], attention_comb.shape[2]]) #reshape to back to orignial dim
    
    #Permutate
    attention_comb = Permute([2, 1])(attention_comb)
    
    sent_representation1 = Multiply()([merged, attention_comb]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*4,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    sent_representation = self.Dropout(sent_representation)
    y = self.d2(sent_representation)



    return y



#For custom training, different input formats
class AttnOnFeatures_ScaleAtt_4grps_withThresFeature2(Model):
  def __init__(self, timesteps, n_feature,n_feature2,n_feature3,n_feature4,n_feature_th,n_feature2_th,n_feature3_th,n_feature4_th,latent_dim):
    super(AttnOnFeatures_ScaleAtt_4grps_withThresFeature2, self).__init__()
    self.latent_dim = latent_dim   
    self.timesteps = timesteps
    self.n_feature = n_feature 
    self.n_feature2 = n_feature2
    self.n_feature3 = n_feature3 
    self.n_feature4 = n_feature4
    self.n_feature_th = n_feature_th
    self.n_feature2_th = n_feature2_th
    self.n_feature3_th = n_feature3_th
    self.n_feature4_th = n_feature4_th
    


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
    
    #For threshold featreus
    self.Masking1_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature_th))
    self.LSTM1_th = LSTM(latent_dim,activation='relu', #tanh
                       kernel_constraint = unit_norm(),
                       recurrent_constraint = max_norm(3),
                       bias_constraint = max_norm(3),
                       activity_regularizer=regularizers.l2(1e-2),
                       kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                       bias_regularizer=regularizers.l2(1e-2),
                       return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking2_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature2_th))
    self.LSTM2_th = LSTM(latent_dim,activation='relu', #tanh
                      kernel_constraint = unit_norm(),
                      recurrent_constraint = max_norm(3),
                      bias_constraint = max_norm(3),
                      activity_regularizer=regularizers.l2(1e-2),
                      kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                      bias_regularizer=regularizers.l2(1e-2),
                      return_sequences=True)
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking3_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature3_th))
    self.LSTM3_th = LSTM(latent_dim,activation='relu', #tanh
                      kernel_constraint = unit_norm(),
                      recurrent_constraint = max_norm(3),
                      bias_constraint = max_norm(3),
                      activity_regularizer=regularizers.l2(1e-2),
                      kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2),
                      bias_regularizer=regularizers.l2(1e-2),
                      return_sequences=True)
    
    #need to use differnt mask and lstm, cuz groups might have differnt n of features, cannot share one lstm layer in the graph
    self.Masking4_th = Masking(mask_value=-1,input_shape=(self.timesteps, self.n_feature4_th))
    self.LSTM4_th = LSTM(latent_dim,activation='relu', #tanh
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
    
    #Attention over time layers
    self.d3 = Dense(1, activation='tanh')
    self.flatten2 = Flatten()
    self.act2 = Activation('softmax')
    self.rv2 = RepeatVector(timesteps) #the size here should be equal to the time steps
    
    #softmax layer for attention norm
    self.norm = Softmax(axis= -1)

  def call(self, input1,input2,input3,input4,input5,input6,input7,input8):
    # #get HPO Sim score
    #    hpo_dist = {"Cardiovascular": 0.0440939,"Metabolism": 0.109443,"Blood": 0.817214,"Respiratory": 0.975282}
    #    TransE_dist = {"Cardiovascular": 0.02150,"Metabolism": 0.33634,"Blood": 0.68064,"Respiratory": 0.97682}

    x1_score = 0.02150
    x2_score = 0.33634
    x3_score = 0.68064
    x4_score = 0.97682
    
    x1 = self.Masking1(input1)
    x1 = self.LSTM1(x1)   
    
    x2 = self.Masking2(input2)
    x2 = self.LSTM2(x2)  
    
    x3 = self.Masking3(input3)
    x3 = self.LSTM3(x3) 
    
    x4 = self.Masking4(input4)
    x4 = self.LSTM4(x4) 
    
    merged = concatenate([x1,x2,x3,x4],axis=2)
    
    #  #Attention On Time
    attention1 = self.d1(merged) # compute importance for each step
    attention = self.flatten(attention1)
    attention = self.act(attention)
    attention = self.rv(attention) #the size here should be equal to hidden size of LSTM1 + LSTM2+...

    #Threshold features
    x1_th = self.Masking1_th(input5)
    x1_th = self.LSTM1_th(x1_th)   
    
    x2_th = self.Masking2_th(input6)
    x2_th = self.LSTM2_th(x2_th)  
    
    x3_th = self.Masking3_th(input7)
    x3_th = self.LSTM3_th(x3_th) 
    
    x4_th = self.Masking4_th(input8)
    x4_th = self.LSTM4_th(x4_th) 
    
    merged2 = concatenate([x1,x2,x3,x4],axis=2)
     
    #Attention on Features
    merged_transp = Permute([2, 1])(merged2)
    attention_f = self.d3(merged_transp) 
    attention_f = self.flatten2(attention_f) 
    attention_f = self.act2(attention_f)
    attention_f = self.rv2(attention_f) #the size here should be equal to hidden size of LSTM1 + LSTM2+...
    attention_f = Permute([2, 1])(attention_f)
    
    #Multiply ontology similarity 
    attention_f1 = attention_f[:,0:8,:]*x1_score
    attention_f2 = attention_f[:,8:16,:]*x2_score
    attention_f3 = attention_f[:,16:32,:]*x3_score
    attention_f4 = attention_f[:,32:48,:]*x4_score
    attention_f5 = concatenate([attention_f1,attention_f2,attention_f3,attention_f4],axis=1)
    

    # #Add or multiply or new rules(Scaled by another one) to combine the two attention
    #attention_comb1 = Multiply()([attention_f5, attention]) #multiply
    #attention_comb = Add()([attention_f5, attention]) #add 
    
    #New rule: time attention scaled by feature attention
    #Find the scale for each rows in feature attention matrix
    v1 = tf.reshape(attention_f5[:,0,:],[tf.shape(attention_f5[:,0,:])[0],1 , tf.shape(attention_f5[:,0,:])[1]]) #get the first row vector
    scale_matrix = Multiply()([attention_f5, 1/v1])  #every raw divided the first row(feature atten)
    attention_comb = Multiply()([attention, scale_matrix]) #scale time att
     
    #TODO: The norm does not work
    #normalize  comb matrix for each sample
    #attention_comb_reshaped = tf.reshape(attention_comb, [tf.shape(attention_comb)[0],tf.shape(attention_comb)[1]*tf.shape(attention_comb)[2]]) #reshape to [sample,all cells for this sample]
    #attention_comb_norm = self.norm(attention_comb_reshaped)
    #attention_comb_norm = tf.reshape(attention_comb_norm, [attention_comb.shape[0],attention_comb.shape[1], attention_comb.shape[2]]) #reshape to back to orignial dim
    
    #Permutate
    attention_comb = Permute([2, 1])(attention_comb)
    
    sent_representation1 = Multiply()([merged, attention_comb]) #multiply merged hidden state with atention weights
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(self.latent_dim*4,))(sent_representation1) #take the some over each time step
    
    # #Maybe add if training for drop out only
    sent_representation = self.Dropout(sent_representation)
    y = self.d2(sent_representation)



    return y
