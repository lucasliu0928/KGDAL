#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:07:18 2021

@author: lucasliu
"""

import tensorflow as tf
import pandas as pd


def representation_dist_between_grps(attended_representation):
    grp1_rep= attended_representation[:,:,0:8]
    grp2_rep = attended_representation[:,:,8:16]
    grp3_rep = attended_representation[:,:,16:24]
    grp4_rep = attended_representation[:,:,24:32]
    
    rep_dist12 = tf.math.reduce_euclidean_norm(grp1_rep-grp2_rep)
    rep_dist13 = tf.math.reduce_euclidean_norm(grp1_rep-grp3_rep)
    rep_dist14 = tf.math.reduce_euclidean_norm(grp1_rep-grp4_rep)
    rep_dist23 = tf.math.reduce_euclidean_norm(grp2_rep-grp3_rep)
    rep_dist24 = tf.math.reduce_euclidean_norm(grp2_rep-grp4_rep)
    rep_dist34 = tf.math.reduce_euclidean_norm(grp3_rep-grp4_rep)
    
    all_rep_dist =  tf.convert_to_tensor([rep_dist12,rep_dist13,rep_dist14,rep_dist23,rep_dist24,rep_dist34])
    #normilze L2
    all_rep_dist_norm = tf.linalg.normalize(all_rep_dist, ord='euclidean', axis=None, name=None)
    
    return all_rep_dist_norm[0]
    
#this should be equal to fit(class_weight), only for training step
def get_sample_weight(class_weight,labels):
   #class_weight = {0: 1, 1: 2}
   sample_weight = labels.numpy().flatten() * class_weight[1] + (1. - labels.numpy().flatten()) * class_weight[0]
   return sample_weight.tolist()

def Train_4grps(model,train_dataset,test_dataset,target_dist,EPOCHS,class_weight,outdir):
    # 
    #model = final_model
    #EPOCHS = 1
    #target_dist = all_transE_dist
    
    #define loss and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    
 
    def loss(model, x1,x2,x3,x4, y,target_dist, training,sample_weight):
      predictions,rep_vector = model(x1,x2,x3,x4, training= training) #returns prediction and vector representation
      y = tf.reshape(y, [-1,1]) #make sure labels  are also 2d ex: [128,1]
      loss_val = loss_object(y_true = y, y_pred = predictions,sample_weight=sample_weight) #make sure labels  are also 2d ex: [128,1]
      
      #Compute the distance between each grp vectors
      rep_vector_dist = representation_dist_between_grps(rep_vector)
      loss_val2 = tf.keras.losses.mean_absolute_error(rep_vector_dist, target_dist) #    #Loss object for  #average mean distance of distance between groups

      return loss_val + loss_val2
    
    def grad(model, x1,x2,x3,x4, y,target_dist,sample_weight):
        with tf.GradientTape() as tape:
            loss_value = loss(model, x1,x2,x3,x4, y,target_dist, training=True,sample_weight=sample_weight)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    #define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy') 
    train_AUC = tf.keras.metrics.AUC()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    test_AUC = tf.keras.metrics.AUC()    
    
    
    #Define check point
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer = optimizer, mod = model)
    manager = tf.train.CheckpointManager(ckpt, outdir +'/tf_ckpts', max_to_keep=100)
    
    # Keep results for plotting
    train_acc_epoch = []
    train_roc_auc_epoch = []
    train_loss_epoch = []
    
    test_acc_epoch = []
    test_roc_auc_epoch = []
    test_loss_epoch = []

        
    for epoch in range(EPOCHS):
          # Reset the metrics at the start of the next epoch
          train_loss.reset_states() #before reset states, the metric function accumativly add prediction results and print the ACC for all samples after for loop
          train_accuracy.reset_states()
          train_AUC.reset_states()
          test_loss.reset_states()
          test_accuracy.reset_states()
          test_AUC.reset_states()
          
          # Iterate over the batches of the dataset.
          for step, (X1,X2,X3,X4,y) in enumerate(train_dataset):
                sample_weight = get_sample_weight(class_weight,y)
                # Optimize the model
                loss_value, grads = grad(model, X1,X2,X3,X4,y,target_dist,sample_weight)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
              
                # Track progress
                train_loss.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                train_pred , _ = model(X1,X2,X3,X4, training= True)
                train_accuracy.update_state(y, train_pred)
                train_AUC.update_state(y,train_pred)
                            
          # End epoch
          train_loss_epoch.append(train_loss.result())
          train_acc_epoch.append(train_accuracy.result())
          train_roc_auc_epoch.append(train_AUC.result())
          
          #Save the weight at the end of each epoch
          save_path = manager.save()
          print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
          ckpt.step.assign_add(1)
          
          ##Run a testing loop at the end of each epoch.
          for test1,test2,test3,test4, test_labels in test_dataset:     
              test_pred , _ = model(test1,test2,test3,test4, training= False)
              #labels = tf.reshape(labels,[tf.shape(labels).numpy()[0],1])
              test_sample_weight = get_sample_weight(class_weight,test_labels)
              loss_value2 = loss(model,test1,test2,test3,test4,test_labels,target_dist,training=False,sample_weight=test_sample_weight)
              
              test_loss(loss_value2)
              test_accuracy.update_state(test_labels, test_pred)
              test_AUC.update_state(test_labels,test_pred)
              
          # End epoch
          test_loss_epoch.append(test_loss.result())
          test_acc_epoch.append(test_accuracy.result())
          test_roc_auc_epoch.append(test_AUC.result())
          
          print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3f} ; Test Loss: {:.3f}, Test Accuracy: {:.3f} ".format(epoch + 1, train_loss.result(), train_accuracy.result(),test_loss.result(),test_accuracy.result()))

    return model, train_acc_epoch, train_loss_epoch, train_roc_auc_epoch,test_acc_epoch,test_loss_epoch,test_roc_auc_epoch

