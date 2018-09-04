from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers

def encoderMod():
	input = Input(shape=(128*128,))
	encoded = Dense(4096, activation='relu')(input)
	encoded = Dense(1024, activation='relu')(encoded)
	encoded = Dense(256, activation='relu')(encoded)
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(16, kernel_regularizer=regularizers.l2(0.01), activation='relu')(encoded)
	encoder = Model(input, encoded)
	return encoder

