#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 00:45:06 2021

@author: lucasliu
"""
import random
from random import choice
import tensorflow as tf

from Model.LSTM_Vanila import VanillaLSTM
from Model.LSTM_ATTonTime import MyLSTM2_4grps
from Train_Eval import Trainer_0318
from Ultility import Evaluation_funcs

def tunning_CustomFit_func(X_train,y_train,X_test,y_test,X_Validation, y_Validation,model_name, timesteps,n_feature,n_epoch,latent_dim, class_weight,opt_lr,batch_size,out_dir):
    # Custom trainer
     
    #Train
    if model_name == "VanillaLSTM":
        #Create a model instance
        tf.random.set_seed(123)
        model = VanillaLSTM(timesteps,n_feature,latent_dim)
        model,train_acc_epoch, train_loss_epoch, train_roc_auc_epoch, test_acc_epoch, test_loss_epoch, test_roc_auc_epoch= Trainer_0318.Trainer_1grp(model,X_train,y_train,X_test,y_test, batch_size, n_epoch,class_weight,opt_lr,out_dir)
    
    #Plot
    Evaluation_funcs.plot_training_history2(n_epoch,train_acc_epoch,test_acc_epoch,train_loss_epoch,test_loss_epoch, out_dir)
    
    #Evalutaion on test data, based on these to choose the final model with optimized paramter
    pred_prob_test = model.predict(X_test, verbose=0)   
    perf_tb_test = Evaluation_funcs.get_all_performce_table(pred_prob_test,y_test,print_res=False)
    test_ACC = perf_tb_test['ACC']
    test_AUC = perf_tb_test['ROC_AUC']
    
    #This can be removed later Evaluation of the model on Validation, just use this to check if the restore is the same as this
    pred_prob = model.predict(X_Validation, verbose=0)    
    #Output model performance table
    perf_tb = Evaluation_funcs.get_all_performce_table(pred_prob,y_Validation,print_res=False)
    perf_tb.to_csv(out_dir + '/perf0320_original.csv')
    
    return test_ACC,test_AUC

def tunning_nonCustomFit_func(train_feature_list,y_train,test_feature_list,y_test,validation_feature_list, y_Validation,model_name, timesteps,n_feature,n_epoch,latent_dim, class_weight,opt_lr,batch_size,out_dir):
    # Custom trainer
    
    #Create a model instance
    if model_name == "VanillaLSTM":
        tf.random.set_seed(123)
        model = VanillaLSTM(timesteps,n_feature,latent_dim)
    elif model_name == "AttonTime":
        tf.random.set_seed(123)
        model = MyLSTM2_4grps(timesteps,n_feature[0],n_feature[1],n_feature[2],n_feature[3],8)
     

    #Train
    opt = tf.keras.optimizers.Adam(learning_rate=opt_lr)
    model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=["acc"])
    # Create a callback that saves the model's weights every 1 epochs
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = out_dir + "/training_2/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,save_freq="epoch")
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    history = model.fit(train_feature_list, y_train, 
                              epochs=n_epoch, batch_size=batch_size,
                              verbose=2, shuffle=True,class_weight = class_weight,
                              callbacks=[cp_callback],
                              validation_data=(test_feature_list, y_test)) #or add manually validation data


    #Plot
    Evaluation_funcs.plot_training_history(history,out_dir)
    
    #Evalutaion on test data, based on these to choose the final model with optimized paramter
    pred_prob_test = model.predict(test_feature_list, verbose=0)   
    perf_tb_test = Evaluation_funcs.get_all_performce_table(pred_prob_test,y_test,print_res=False)
    test_ACC = perf_tb_test['ACC']
    test_AUC = perf_tb_test['ROC_AUC']
    
    #This can be removed later Evaluation of the model on Validation, just use this to check if the restore is the same as this
    pred_prob = model.predict(validation_feature_list, verbose=0)    
    #Output model performance table
    perf_tb = Evaluation_funcs.get_all_performce_table(pred_prob,y_Validation,print_res=False)
    perf_tb.to_csv(out_dir + '/perf0320_original.csv')
    
    return test_ACC,test_AUC

def generate_random_hyper_parameter(latent_dim_list,batch_size_list,opt_lr_list,num_exp):
    random.seed(321)
    random_parameters = []
    for i in range(num_exp):
        random_parameters.append([choice(latent_dim_list), choice(batch_size_list), choice(opt_lr_list)])
    return random_parameters