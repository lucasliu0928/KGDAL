#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:32:12 2021

@author: lucasliu
"""

import tensorflow as tf

def Load_RandomCases(n_train_sample,n_test_sample,n_valid_sample,batch_size, n_featureA, n_featureB, timesteps):
    # n_train_sample = 200
    # n_test_sample = 50
    # n_valid_sample = 10
    # batch_size = 32
    # timesteps = 12
    # n_featureA = 6
    # n_featureB = 5
    
    tf.random.set_seed(1)
    ##Generate random training samples (#sample,timestep,feature)
    X_train_A = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_train_B = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    #featuregrpA[0,1,:] = -1
    y_train = tf.random.categorical(tf.math.log([[0.7, 0.3]]), n_train_sample).numpy()
    y_train = y_train.transpose()
    print(X_train_A.shape,X_train_B.shape)
    
    
    ##Generate random testing samples (#sample,timestep,feature)
    X_test_A = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_test_B = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    #featuregrpA[0,1,:] = -1
    y_test = tf.random.categorical(tf.math.log([[0.5, 0.5]]), n_test_sample).numpy()
    y_test = y_test.transpose()
    print(X_test_A.shape,X_test_B.shape)
    
    
    ##Generate random validation samples (#sample,timestep,feature)
    X_Validation_A = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_Validation_B = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    #featuregrpA[0,1,:] = -1
    y_Validation = tf.random.categorical(tf.math.log([[0.5, 0.5]]), n_valid_sample).numpy()
    y_Validation = y_Validation.transpose()
    print(X_Validation_A.shape,X_Validation_B.shape)
    
    
    
    # Prepare the batches training dataset.
    tf.random.set_seed(1234)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_A,X_train_B, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=X_train_A.shape[0]).batch(batch_size)
    
    # Prepare the batches test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_A,X_test_B, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    #validation 
    #validation_dataset = tf.data.Dataset.from_tensor_slices((X_Validation_A,X_Validation_B, y_Validation))

    
    return train_dataset,test_dataset,X_Validation_A,X_Validation_B,y_Validation


def Load_RandomCases_4grps(n_train_sample,n_test_sample,n_valid_sample,batch_size, n_featureA, n_featureB,n_featureC,n_featureD, timesteps):
    # n_train_sample = 200
    # n_test_sample = 50
    # n_valid_sample = 10
    # batch_size = 32
    # timesteps = 12
    # n_featureA = 6
    # n_featureB = 5    
    # n_featureC = 6
    # n_featureD = 5
    
    tf.random.set_seed(1)
    ##Generate random training samples (#sample,timestep,feature)
    X_train_A = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_train_B = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    X_train_C = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureC], minval=1,maxval=10).numpy() 
    X_train_D = tf.random.uniform(shape = [n_train_sample,timesteps,n_featureD], minval=1,maxval=10).numpy() 

    #featuregrpA[0,1,:] = -1
    y_train = tf.random.categorical(tf.math.log([[0.7, 0.3]]), n_train_sample).numpy()
    y_train = y_train.transpose()
    print(X_train_A.shape,X_train_B.shape,X_train_C.shape,X_train_D.shape)
    
    
    ##Generate random testing samples (#sample,timestep,feature)
    X_test_A = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_test_B = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    X_test_C = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureC], minval=1,maxval=10).numpy() 
    X_test_D = tf.random.uniform(shape = [n_test_sample,timesteps,n_featureD], minval=1,maxval=10).numpy() 

    #featuregrpA[0,1,:] = -1
    y_test = tf.random.categorical(tf.math.log([[0.5, 0.5]]), n_test_sample).numpy()
    y_test = y_test.transpose()
    print(X_test_A.shape,X_test_B.shape,X_test_C.shape,X_test_D.shape)
    
    
    ##Generate random validation samples (#sample,timestep,feature)
    X_Validation_A = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureA], minval=1,maxval=10).numpy()
    X_Validation_B = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureB], minval=1,maxval=10).numpy() 
    X_Validation_C = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureC], minval=1,maxval=10).numpy() 
    X_Validation_D = tf.random.uniform(shape = [n_valid_sample,timesteps,n_featureD], minval=1,maxval=10).numpy() 


    #featuregrpA[0,1,:] = -1
    y_Validation = tf.random.categorical(tf.math.log([[0.5, 0.5]]), n_valid_sample).numpy()
    y_Validation = y_Validation.transpose()
    print(X_Validation_A.shape,X_Validation_B.shape,X_Validation_C.shape,X_Validation_D.shape)
    
    
    
    # Prepare the batches training dataset.
    tf.random.set_seed(1234)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_A,X_train_B,X_train_C,X_train_D, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=X_train_A.shape[0]).batch(batch_size)
    
    # Prepare the batches test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_A,X_test_B,X_test_C,X_test_D, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    #validation 
    #validation_dataset = tf.data.Dataset.from_tensor_slices((X_Validation_A,X_Validation_B, y_Validation))

    
    return train_dataset,test_dataset,X_Validation_A,X_Validation_B,X_Validation_C,X_Validation_D,y_Validation