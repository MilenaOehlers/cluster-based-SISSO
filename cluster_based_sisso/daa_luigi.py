#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:31:47 2020
matplotlib==3.2.1
tensorflow==2.3.1
tensorflow-estimator==2.3.0
tensorflow-probability==0.9.0

@author: sbailò, oehlers
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
import pandas as pd
from func_collection import get_zfixed

def build_network(intermediate_dim = 4, batch_size = 1024, latent_dim = 2, epochs = 100):
    
    def execute(data, at_loss_factor=8.0, target_loss_factor=8.0,recon_loss_factor=4.0,kl_loss_factor=4.0):
        #We use variational autoencoders to map the data set into a latent space. The neural network is constructed to force data in latent space to be defined within an arbitrary convex hull. We use a triangular convex hull as shown below. 
        
        zfixed = get_zfixed (2)
        
        #We construct a variational autoencoder that generates a mean $\mu$, and a standard deviation $\sigma$ for each data point. The point is then mapped into the latent space with a stochastic extraction from a Gaussian $\mathcal{N}(\mu,\,\sigma^{2})$, where $\mu$'s are by construction within a hull $z_{fixed}$. 
        x_train = data['train_feat']
        y_train = data['train_targets']
        x_test = data['test_feat']
        y_test = data['test_targets']
        
        original_dim = x_train.shape [1]
        try:
            sideinfo_dim = y_train.shape [1]
        except:
            sideinfo_dim = 1
        
        x_train = np.array(np.reshape(x_train, [-1, original_dim]), dtype='float32')
        y_train = np.array(np.reshape(y_train, [-1, sideinfo_dim]), dtype='float32')
        
        x_test = np.array(np.reshape(x_test, [-1, original_dim]), dtype='float32')
        y_test = np.array(np.reshape(y_test, [-1, sideinfo_dim]), dtype='float32')
        
        
        # network parameters
        simplex_vrtxs = latent_dim + 1
        
        
        # encoder
        input_x = tfk.Input(shape=(original_dim,), name='encoder_input_x', dtype='float32')
        
        x = tfkl.Dense(intermediate_dim, activation='relu')(input_x)
        x = tfkl.Dense(intermediate_dim, activation='relu')(x)
        A = tfkl.Dense(simplex_vrtxs, activation='linear')(x)
        A = tfkl.Dense(simplex_vrtxs, activation=tf.nn.softmax)(A)
        B_t = tfkl.Dense(simplex_vrtxs, activation='linear')(x)
        B = tf.nn.softmax(tf.transpose(B_t), axis=1)
        
        z_fixed = get_zfixed (latent_dim)
        z_fixed = tf.constant (z_fixed, dtype='float32')
        mu = tf.matmul(A, z_fixed)
        z_pred = tf.matmul(B,mu)
        sigma = tfkl.Dense(latent_dim)(x)
        t = tfd.Normal(mu,sigma)
        
        input_y = tfk.Input(shape=(sideinfo_dim,), name='encoder_input_y', dtype='float32')
        y = tf.identity(input_y)
        
        encoder = tfk.Model([input_x,input_y], [t.sample(),A,mu,sigma, tf.transpose(B) ,y], name='encoder')
        encoder.summary()
        
        # decoder
        latent_inputs = tfk.Input(shape=(latent_dim,), name='z_sampling')
        input_y_lat = tfk.Input(shape=(sideinfo_dim,), name='encoder_input_y_lat')
        
        x = tfkl.Dense(intermediate_dim, activation='relu')(latent_inputs)
        x = tfkl.Dense(original_dim, activation='linear')(x)
        x_hat = tfkl.Dense(original_dim, activation='linear')(x)
        
        y = tfkl.Dense(intermediate_dim, activation='relu')(latent_inputs)
        y = tfkl.Dense(intermediate_dim, activation='relu')(y)
        y_hat = tfkl.Dense(sideinfo_dim, activation='linear')(y) 
        
        decoder = tfk.Model([latent_inputs,input_y_lat], [x_hat,y_hat], name='decoder')
        decoder.summary()

        # VAE
        encoded = encoder([input_x,input_y])
        outputs = decoder([encoded[0],encoded[-1]])
        vae = tfk.Model([input_x,input_y], outputs, name='vae')
        
        reconstruction_loss = tfk.losses.mse (input_x, outputs[0])
        class_loss = tfk.losses.mse ( input_y, outputs[1])
        archetype_loss = tf.reduce_sum( tfk.losses.mse(z_fixed, z_pred))
        
        kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        # annealing kl_loss parameter (milena):
        anneal = 0
        kl_loss_max = kl_loss_factor
        if anneal == 1:
            kl_loss_factor = tfk.backend.variable(0.)
            class NewCallback(tfk.callbacks.Callback):
                def __init__(self, kl_loss_factor):
                    self.kl_loss_factor = kl_loss_factor
                def on_epoch_end(self, epoch, logs={}):
                    if epoch <= 100:
                        tfk.backend.set_value(self.kl_loss_factor, tfk.backend.get_value(self.kl_loss_factor) + epoch/100*kl_loss_max)

        callbacks = [NewCallback(kl_loss_factor),] if anneal == 1 else None # milena
        
        vae_loss = tf.reduce_mean(recon_loss_factor*reconstruction_loss
                                  + target_loss_factor*class_loss 
                                  + kl_loss_factor*kl_loss 
                                  + at_loss_factor*archetype_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
        
        vae.fit([x_train,y_train],
                epochs=epochs,
                batch_size=batch_size,
                validation_split = 0.25,
                callbacks = callbacks)
        
        # archetypes
        archetypes,_ = decoder ([z_pred, tf.zeros([3,3])])
        get_archtypes = tfk.Model (input_x, [archetypes,z_pred] , name='get_zpred')
        
        
        t,A,mu,sigma, B_t, y = encoder.predict([x_train,np.zeros(np.shape(y_train))])
        archetypes_pred, z_pred = get_archtypes(x_train)
        
        x_train_pred, y_train_pred = vae.predict([x_train,np.zeros(np.shape(y_train))])
        t_train,A_train,mu_train,sigma_train, B_t_train, y_trainzeros = encoder.predict([x_train,np.zeros(np.shape(y_train))])
        
        x_test_pred, y_test_pred = vae.predict([x_test,np.zeros(np.shape(y_test))])
        t_test,A_test,mu_test,sigma_test, B_t_test, y_testzeros = encoder.predict([x_test,np.zeros(np.shape(y_test))])

        result = {('train','real space','features'): x_train,
                  ('train','real space', 'targets'): y_train,
                  ('train', 'latent space', 'As'): A_train,
                  ('train','latent space','mus'): mu_train,
                  ('train','latent space','sigmas'): sigma_train,
                  ('train','reconstructed real space','features'): x_train_pred,
                  ('train','reconstructed real space','targets'): y_train_pred,
                  ('test', 'real space', 'features'): x_test,
                  ('test', 'real space', 'targets'): y_test,
                  ('test', 'latent space', 'As'): A_test,
                  ('test', 'latent space', 'mus'): mu_test,
                  ('test', 'latent space', 'sigmas'): sigma_test,
                  ('test', 'reconstructed real space', 'features'): x_test_pred,
                  ('test', 'reconstructed real space', 'targets'): y_test_pred }
        

        return result
    
    return execute
