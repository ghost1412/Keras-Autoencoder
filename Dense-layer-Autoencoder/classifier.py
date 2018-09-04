'''
		-------------------------
---------------|  importing libraries    |-------------------------------------------------------
		--------------------------
'''
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
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
from keras import regularizers
from keras.layers.advanced_activations import PReLU
from encoderMod import encoderMod
from decoderMod import decoderMod
'''
		-------------------------
|---------------|    conv autoencoder	 |------------------------------------------------------|
		--------------------------
'''
encoder = encoderMod()
encoder.summary()

decoder = decoderMod()
decoder.summary()

input = Input(shape=(128*128,))
outputs = decoder(encoder(input))

autoencoder = Model(input, outputs)
autoencoder.summary()
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

'''
		-------------------------------------------------------------------------
|---------------| loading pretrained weights of autoencoder and freezing autoenc layers	 |-------------------------|
		|			transferlearning				 |	 
		-------------------------------------------------------------------------
'''
autoencoder.load_weights("deepconvo1reg.h5")
print("Loaded Model from disk")
autoencoder.layers[0].trainable = False
autoencoder.layers[1].layers[0].trainable = False
autoencoder.layers[1].layers[1].trainable = False
autoencoder.layers[1].layers[2].trainable = False
autoencoder.layers[1].layers[3].trainable = False
autoencoder.layers[1].layers[4].trainable = False
autoencoder.layers[1].layers[5].trainable = False
#autoencoder.layers[9].trainable = False
#autoencoder.layers[10].trainable = False
autoencoder.layers.pop()

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
autoencoder.summary()


'''
		-------------------------
|---------------|     preparing data	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn.model_selection import train_test_split

reader = csv.reader(open("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data[15000:32000]).astype("str")    #read data in string form
print(result.shape)

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
labels = result[0: , -1]                                      #label for classification
#one_hot_labels = to_categorical(labels, num_classes=2)
result_train = result[0: , 1:len(result[1])-6]                #sliceing last and 1st coloumn out
result = None
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
input = np.empty((len(result_train), 128, 128))             #features
x_train_extra = np.array(result_train[0:, -6:-2])	      #extra attributes
for j in range(0, len(result_train)):                         #input to classifer to 128*128
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1310), 'constant'),(128,128))
  input[j] = resultraintemp
result_train = None
input = np.reshape(input, [-1, 128*128])
#input1 = np.reshape(input, [-1, 128*128])

'''import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
X_tsne = TSNE(learning_rate=100).fit_transform(input1)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
#plt.subplot(122)
plt.savefig('aa')'''


'''		-------------------------
|---------------|     latent layer	 |--------------------------------------------------------|
		-------------------------
'''
layer_name = 'model_1'

intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.layers[1].get_output_at(1))
intermediate_layer_model.summary()
latent_layer = np.array(intermediate_layer_model.predict(input))
input = None


#mearging extra attributes
s = np.concatenate((latent_layer, x_train_extra), axis=1)
print(latent_layer.shape)
#s = np.concatenate((s, x_train_extra), axis=1)

'''import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
X_tsne = TSNE(learning_rate=100).fit_transform(s)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
#plt.subplot(122)
plt.savefig('bb')'''



# Split our data
x_train, x_test, y_train, y_test = train_test_split(s,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=42)
labels = None 
s = None
print(x_train.shape)
from sklearn.metrics import classification_report, confusion_matrix
  
'''		-------------------------
|---------------|     naive bayes	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
print(x_train_extra[0,-1])
from sklearn import preprocessing
x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
gnb = GaussianNB()
print('bayes')
# Train tclassifier
model = gnb.fit(x_train, y_train)

# Make predictions
y_pred = gnb.predict(x_test)
#print(preds)

# Evaluate accuracy
print('bayes:')
print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 

'''		-------------------------
|---------------|     logistic reg	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
y_pred = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('logestic reg :')
print(score)
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 

'''		-------------------------
|---------------|     decision tree	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn import tree
from sklearn.model_selection import train_test_split
clf = tree.DecisionTreeClassifier()
clf.fit(X=x_train, y=y_train)
clf.feature_importances_ # [ 1.,  0.,  0.]
y_pred = clf.predict(x_test)
acc = clf.score(X=x_test, y=y_test) # 1.0
print('decision tree')
print(acc)
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 
 

'''		-------------------------
|---------------|     svm radical	 |--------------------------------------------------------|
'''

from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(x_train, y_train)  
y_pred = svclassifier.predict(x_test) 
print('svm radical') 
print(svclassifier.score(x_test, y_test))
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred))  

'''		-------------------------
|---------------|     zeroR		 |--------------------------------------------------------|
		-------------------------
'''

from sklearn.dummy import DummyClassifier 
zeroRclassifier = DummyClassifier()
zeroRclassifier.fit(x_train, y_train)  
y_pred = zeroRclassifier.predict(x_test) 
print('zeroR') 
print(zeroRclassifier.score(x_test, y_test))
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 

'''		-------------------------
|---------------|     svm linear	 |--------------------------------------------------------|
		-------------------------
'''

from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train)  
y_pred = svclassifier.predict(x_test) 
print('svm linear') 
print(svclassifier.score(x_test, y_test))
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 
