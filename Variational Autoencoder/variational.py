'''
		-------------------------
---------------|  importing libraries    |-------------------------------------------------------
		--------------------------
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from sklearn.preprocessing import StandardScaler
import argparse
'''
		-------------------------
---------------|    reading 	csv	 |-------------------------------------------------------
		--------------------------
'''
reader = csv.reader(open("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data[1:2500]).astype("str")    #read data in string form
print(result.shape)
feature_names = result[0, 1 : len(result[1])-1]

#removing unwanted strings butrans = 0 and opana = 1 and empty cells with 0 and making 1st cell equal 0 so#
for j in range(0, len(result)):
  result[j][0] = '0'
  for i in range(0, len(result[j])):
    if(result[j][i] == "butrans"):
      result[j][i] = '7'
    if(result[j][i] == "opana"):
      result[j][i] = '8'
    if(result[j][i] == ''):
      result[j][i] = '0'
    if(result[j][i] == "Butrans and Opana"):
      result[j][i] = '9' 
    if(result[j][i] == 'Frequent'):
      result[j][i] = '0' 
    if(result[j][i] == 'Non Frequent'):
      result[j][i] = '1'

labels = result[1: , -2]                                      #label for classification
result_train = result[0: , 1:len(result[1])-2]              #training data for autoencoder
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
x_train = np.empty((len(result_train), 128, 128,1))            

for j in range(0, len(result_train)):                         #reshaping to 2d array for convolutional autoencoder
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1306), 'constant'),(128,128,1))
  x_train[j] = resultraintemp

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

  
'''
		-------------------------
---------------| variational-autoencoder |-------------------------------------------------------
		--------------------------
'''
original_dim = x_train.shape[1] * x_train.shape[1]
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 7
latent_dim = 2
epochs = 500

x_train = np.reshape(x_train, [-1, original_dim])
x_test = x_train/2
#x_train = StandardScaler().fit_transform(x_train)
# VAE model = encoder + decoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
#x = BatchNormalization()
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', metrics=['accuracy'])
vae.summary()
vae.fit(x_train,
        epochs=epochs,
       	batch_size=batch_size,
       	validation_data=(x_test, None))
vae.save_weights('vae_mlp_mnist.h5')
