#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:53:45 2021

@author: lucasliu
"""

import tensorflow as tf

#this should be equal to fit(class_weight), only for training step
def get_sample_weight(class_weight,labels):
   #class_weight = {0: 1, 1: 2}
   sample_weight = labels.numpy().flatten() * class_weight[1] + (1. - labels.numpy().flatten()) * class_weight[0]
   return sample_weight.tolist()

def Trainer_1grp(input_model,X_train,y_train,X_test,y_test, batch_size, EPOCHS,class_weight,lr,outdir):
     # Prepare the batches training dataset.
    tf.random.set_seed(1234)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=30000).batch(batch_size)
    #train_dataset  = LoadModelReady_data.create_balanced_batches(X_train,y_train,batch_size)
              
    # Prepare the batches test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    #test_dataset = LoadModelReady_data.create_balanced_batches(X_test,y_test,batch_size)
            
        
    #define loss and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    #define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy') 
    train_AUC = tf.keras.metrics.AUC()


    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    test_AUC = tf.keras.metrics.AUC()    
    
    #@tf.function
    def train_step(data , labels,sample_weight):
      with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = input_model(data, training=True)
        #loss = loss_object(labels, predictions) #make sure labels  are also 2d ex: [128,1]
        labels = tf.reshape(labels,[tf.shape(labels).numpy()[0],1])
        loss = loss_object(labels, predictions,sample_weight=sample_weight) #make sure labels  are also 2d ex: [128,1]

      gradients = tape.gradient(loss, input_model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, input_model.trainable_variables))
    
      train_loss(loss)
      train_accuracy(labels, predictions)
      train_AUC(labels,predictions)

      
    #@tf.function
    def test_step(data, labels,sample_weight):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = input_model(data, training=False)
      labels = tf.reshape(labels,[tf.shape(labels).numpy()[0],1])
      t_loss = loss_object(labels, predictions,sample_weight=sample_weight)
    
      test_loss(t_loss)
      test_accuracy(labels, predictions)
      test_AUC(labels,predictions)

    #Define check point
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer = optimizer, mod = input_model)
    manager = tf.train.CheckpointManager(ckpt, outdir +'/tf_ckpts', max_to_keep=100)
    
    ##Start training
    #EPOCHS = 5
    test_acc_epoch = []
    test_roc_auc_epoch = []
    test_loss_epoch = []
    train_acc_epoch = []
    train_roc_auc_epoch = []
    train_loss_epoch = []
    
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states() #before reset states, the metric function accumativly add prediction results and print the ACC for all samples after for loop
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()
      
      # Iterate over the batches of the dataset.
      for step, (X,y) in enumerate(train_dataset):
          sample_weight = get_sample_weight(class_weight,y)
          train_step(X,y,sample_weight)
        
      #Save the weight at the end of each epoch
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      ckpt.step.assign_add(1)
    
      #Run a testing loop at the end of each epoch.
      for test_data, test_labels in test_dataset:  
         sample_weight = get_sample_weight(class_weight,test_labels)
         test_step(test_data, test_labels,sample_weight)
      
      test_loss_epoch.append(test_loss.result().numpy())
      test_roc_auc_epoch.append(test_AUC.result().numpy())       
      test_acc_epoch.append(test_accuracy.result().numpy())
      train_loss_epoch.append(train_loss.result().numpy())
      train_roc_auc_epoch.append(train_AUC.result().numpy())       
      train_acc_epoch.append(train_accuracy.result().numpy())
      ################################
      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():.3f}, '
        f'Accuracy: {train_accuracy.result() * 100:.3f}, '
        f'AUC: {train_AUC.result() * 100:.3f}, '
        f'Test Loss: {test_loss.result():.3f}, '
        f'Test Accuracy: {test_accuracy.result() * 100:.3f}'
        f'Test AUC: {test_AUC.result() * 100:.3f}'
      )
      
    return input_model,train_acc_epoch, train_loss_epoch, train_roc_auc_epoch, test_acc_epoch, test_loss_epoch, test_roc_auc_epoch

def Train_4grps(model,train_dataset,test_dataset,EPOCHS,class_weight,outdir):
    #define loss and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    
    
    def loss(model, x1,x2,x3,x4, y, training,sample_weight):
      predictions = model(x1,x2,x3,x4, training= training)
      loss_val = loss_object(y_true = y, y_pred = predictions,sample_weight=sample_weight) #make sure labels  are also 2d ex: [128,1]
      return loss_val
    
    def grad(model, x1,x2,x3,x4, targets,sample_weight):
        with tf.GradientTape() as tape:
            loss_value = loss(model, x1,x2,x3,x4, targets, training=True,sample_weight=sample_weight)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    
    #define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy') 
    train_AUC = tf.keras.metrics.AUC()


    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    # test_AUC = tf.keras.metrics.AUC()    
    
    
    #Define check point
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer = optimizer, mod = model)
    manager = tf.train.CheckpointManager(ckpt, outdir +'/tf_ckpts', max_to_keep=100)
    
    # Keep results for plotting
    train_acc_epoch = []
    train_roc_auc_epoch = []
    train_loss_epoch = []
    
    # test_acc_epoch = []
    # test_roc_auc_epoch = []
    # test_loss_epoch = []

        
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states() #before reset states, the metric function accumativly add prediction results and print the ACC for all samples after for loop
      train_accuracy.reset_states()
      train_AUC.reset_states()
      # test_loss.reset_states()
      # test_accuracy.reset_states()
      
      # Iterate over the batches of the dataset.
      for step, (X1,X2,X3,X4,y) in enumerate(train_dataset):
            sample_weight = get_sample_weight(class_weight,y)
            # Optimize the model
            loss_value, grads = grad(model, X1,X2,X3,X4,y,sample_weight)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
          
            # Track progress
            train_loss.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            train_pred = model(X1,X2,X3,X4, training= True)
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
      
      if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    train_loss.result(),
                                                                    train_accuracy.result()))
      
    # for epoch in range(EPOCHS):
    #   #Run a testing loop at the end of each epoch.
    #   for test_data1,test_data2,test_data3,test_data4, test_labels in test_dataset:     
    #      test_step(test_data1, test_data2,test_data3,test_data4, test_labels)
      
    #   test_loss_epoch.append(test_loss.result().numpy())
    #   test_roc_auc_epoch.append(test_AUC.result().numpy())       
    #   test_acc_epoch.append(test_accuracy.result().numpy())
    #   train_loss_epoch.append(train_loss.result().numpy())
    #   train_roc_auc_epoch.append(train_AUC.result().numpy())       
    #   train_acc_epoch.append(train_accuracy.result().numpy())
    #   ################################
    #   print(
    #     f'Epoch {epoch + 1}, '
    #     f'Loss: {train_loss.result():.3f}, '
    #     f'Accuracy: {train_accuracy.result() * 100:.3f}, '
    #     f'AUC: {train_AUC.result() * 100:.3f}, '
    #     f'Test Loss: {test_loss.result():.3f}, '
    #     f'Test Accuracy: {test_accuracy.result() * 100:.3f}'
    #     f'Test AUC: {test_AUC.result() * 100:.3f}'
    #   )

    return model, train_acc_epoch, train_loss_epoch, train_roc_auc_epoch


