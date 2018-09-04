from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.advanced_activations import PReLU

def decoderMod():
	latent_inputs = Input(shape=(16,))
	decoded = Dense(64, activation='relu')(latent_inputs)
	decoded = Dense(256, activation='relu')(decoded)
	decoded = Dense(1024, activation='relu')(decoded)
	decoded = Dense(4096, activation='relu')(decoded)
	decoded = Dense(128*128, activation='sigmoid')(decoded)
	decoder = Model(latent_inputs, decoded)
	return decoder

