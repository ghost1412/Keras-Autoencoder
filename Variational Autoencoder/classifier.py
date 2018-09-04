'''
		-------------------------
---------------|  importing libraries    |-------------------------------------------------------
		--------------------------
'''
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Dense, Conv2D, Lambda, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K
from statistics import mean
import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
from keras.losses import mse, binary_crossentropy
'''
		---------------------------------
|---------------|        Variational auto	 |-------------------------------------------------------|
		---------------------------------
'''

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

original_dim = 128 * 128
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 8
latent_dim = 2
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# buidecoder = Model(latent_inputs, outputs, name='decoder')ld decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
#decoder.summary()
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


'''
		-------------------------------------------------------------------------
|---------------| loading pretrained weights of autoencoder and freezing autoenc layers	 |-------------------------|
		|			transferlearning				 |	 
		-------------------------------------------------------------------------
'''
#vae.summary()
vae.load_weights("vae_mlp_mnist.h5")
print("Loaded Model from disk")
#print(encoded[1])
vae.layers[0].trainable = False
vae.layers[1].trainable = False
vae.layers.pop()
vae.compile(optimizer='adam', metrics=['accuracy'])
#vae.summary()

'''
		---------------------------------
|---------------|        classifier		 |-------------------------------------------------------|
		---------------------------------
'''

latent_inputs_cly = Input(shape=(latent_dim,), name='z_sampling_cly')
xx = Dense(intermediate_dim, activation='relu')(latent_inputs_cly)
xx = Dense(2, activation='relu')(xx)
clasifier_layer = Model(latent_inputs_cly, xx, name='classifier')
xx = clasifier_layer(encoder(inputs)[2])
clasifier = Model(inputs, xx, name='vae_mlp1')
clasifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
clasifier.summary()

'''
		-------------------------
|---------------|     preparing data	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn.model_selection import train_test_split

reader = csv.reader(open("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data[1:2500]).astype("str")    #read data in string form
print(result.shape)

#removing unwanted strings butrans = 0 and opana = 1 and empty cells with 0 and making 1st cell equal 0 so#
for j in range(0, len(result)):
  result[j][0] = '0'
  for i in range(0, len(result[j])):
    if(result[j][i] == "butrans"):
      result[j][i] = '8'
    if(result[j][i] == "opana"):
      result[j][i] = '9'
    if(result[j][i] == ''):
      result[j][i] = '0'
    if(result[j][i] == "Butrans and Opana"):
      result[j][i] = '7' 
    if(result[j][i] == 'Frequent'):
      result[j][i] = '0' 
    if(result[j][i] == 'Non Frequent'):
      result[j][i] = '1'
labels = result[1: , -1]                                      #label for classification
one_hot_labels = to_categorical(labels, num_classes=2)
result_train = result[1: , 1:len(result[1])-2]                #sliceing last and 1st coloumn out
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
input = np.empty((len(result_train), 128, 128,1))             #features

for j in range(0, len(result_train)):                         #input to classifer to 128*128
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1306), 'constant'),(128,128,1))
  input[j] = resultraintemp
input = np.reshape(input, [-1, original_dim])
#train_labels_one_hot  = np.empty((len(result), 2, 1,))
#hot_encoding = np.zeros((2, 1))
#for i in range(0, len(result)):
 #   if(result[i][-1] == 1):
  #      hot_encoding[1] = 1
  #  else:
   #     hot_encoding[0] = 0
   # train_labels_one_hot[i] = hot_encoding 

'''
		-------------------------
|---------------|  training classifier	 |--------------------------------------------------------|
		-------------------------
'''
from keras.callbacks import TensorBoard
input = input/2
x_train, x_test, y_train, y_test = train_test_split(input,
                                                          one_hot_labels,
                                                          test_size=0.33,
                                                          random_state=42)

#classifier.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
history = clasifier.fit(x_train, y_train,
                	 epochs=100,
               		 batch_size=6,
               		 shuffle=True,
               		 validation_data=(x_test, y_test),
               		 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
clasifier.predict(x_test)
print(mean(history.history['acc']))
print(mean(history.history['val_acc']))

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(history.history['acc'], history.history['val_acc'])
fig.savefig('/home/tonystark/Downloads/gh/')   # save the figure to file
plt.close(fig)    # close the figure
