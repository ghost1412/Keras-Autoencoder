from encoderMod import encoderMod
from decoderMod import decoderMod
from dataprocess import *
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.callbacks import CSVLogger
from keras import regularizers

def autoencoder():
	encoder = encoderMod()
	encoder.summary()
	decoder = decoderMod()
	decoder.summary()
	input = Input(shape=(128*128,))
	#print(encoder(input)[3])
	outputs = decoder(encoder(input))
	autoencoder = Model(input, outputs)
	return autoencoder

def custom_loss(input, output):
	alpha = 0.5 
	reconstruction_loss = mse(input, output)
#	reconstruction_loss *= 128*128
	reconstruction_loss *= alpha
	relation_loss = mse(K.dot(K.transpose(input), input),K.dot(K.transpose(output), output))
#	relation_loss *= 128*128
	relation_loss *= (1-alpha)
	auto_loss = reconstruction_loss + relation_loss
	return reconstruction_loss

def compiler(autoencoder):
	autoencoder.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='mse', metrics=['accuracy'])
	return autoencoder

def Fit(autoencoder, x_trian, x_test, epochs, batch_size):
	history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))
	return history

def saveWeight(model):
	model.save_weights("deepconvo1reg.h5")
	
if __name__ == "__main__":
	autoencoder = autoencoder()
	autoencoder = compiler(autoencoder)
	data = read_csv("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv")
	labels, result_train = process_data(data, 10000)
	x_train = reshape(result_train)
	x_train, x_test, y_train, y_test = split_train_test(x_train, labels)
	x_train, x_test = datapreprocess(x_train, x_test)
	history = Fit(autoencoder, x_train, x_test, 50, 8)
	saveWeight(autoencoder)

	import matplotlib
	matplotlib.use('agg')
	from matplotlib import pyplot as plt
	plt.switch_backend('agg')
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('ddreg')
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('eereg')



	loss_history = history.history["loss"]
	import numpy
	numpy_loss_history = numpy.array(loss_history)
	numpy.savetxt("loss_historyreg.txt", numpy_loss_history, delimiter=",")
	loss_accuracy = history.history["acc"]
	numpy_loss_accuracy = numpy.array(loss_accuracy)
	numpy.savetxt("loss_accuracyreg.txt", numpy_loss_accuracy, delimiter=",")
	loss_val_history = history.history["val_loss"]
	import numpy
	numpy_val_loss_history = numpy.array(loss_val_history)
	numpy.savetxt("val_loss_historyreg.txt", numpy_val_loss_history, delimiter=",")
	val_loss_accuracy = history.history["val_acc"]
	numpy_val_loss_accuracy = numpy.array(val_loss_accuracy)
	numpy.savetxt("val_loss_accuracyreg.txt", numpy_val_loss_accuracy, delimiter=",")

