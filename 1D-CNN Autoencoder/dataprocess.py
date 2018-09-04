import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def read_csv(path):
	reader = csv.reader(open(path, "r"), delimiter=",")
	data = list(reader)
	return data

def process_data(data, length):
	result = np.array(data[1:length]).astype("str")
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
	labels = result[0: , -1] 
	result_train = result[0: , 1:len(result[1])-6]     
	return labels, result_train

def reshape(result_train):
	result_train = np.array(result_train).astype(np.float)        
	x_train = np.empty((len(result_train), 128*128))            
	for j in range(0, len(result_train)):                         
	  x_train[j] = np.reshape(np.pad(result_train[j], (0, 1310), 'constant'),(128*128)) 
	return x_train   

def split_train_test(x_train, labels):
	x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                          labels,
                                                          test_size=0.30,
                                                          random_state=42)
	return x_train, x_test, y_train, y_test

def datapreprocess(x_train, x_test):
	return preprocessing.scale(x_train), preprocessing.scale(x_test)
