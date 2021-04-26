#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:07:17 2021

@author: lucasliu
"""
import tensorflow as tf
from Model.LSTM_ATTonTime import MyLSTM2_4grps
from Model.LSTM_ATTonTimeAndFeature_WithThresFeatures import AttnOnFeatures_ScaleAtt_4grps_withThresFeature
from Model.LSTM_Vanila import VanillaLSTM

def create_model_instance(timesteps,n_feature,latent_dim,model_name):
    #Create an instance of the model 
    if model_name == "VanillaLSTM":
        tf.random.set_seed(123)
        reconstructed_model = VanillaLSTM(timesteps,n_feature,latent_dim)
    elif model_name == "AttonTime":
        tf.random.set_seed(123)
        reconstructed_model = MyLSTM2_4grps(timesteps,n_feature[0],n_feature[1],n_feature[2],n_feature[3],8)
    return reconstructed_model

def restore_model(reconstructed_model,check_point_file,training_type):
    if training_type == "CustomFit": #Restore from custom training
        #check_point_file = "/tf_ckpts/ckpt-100"
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), mod = reconstructed_model)
        ckpt.restore(check_point_file).expect_partial() #use expect_partial to only restore vars used for validation, removed warnings
    elif training_type == "nonCustomFit": #Restore from non-custom training
        #check_point_file = "/training_2/cp-0002.ckpt"
        reconstructed_model.load_weights(check_point_file)

    return reconstructed_model
