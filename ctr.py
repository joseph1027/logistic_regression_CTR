from scipy.sparse import *
from scipy import *
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Input,Embedding,Dense,Dropout,Activation,Add,concatenate
from keras import optimizers
from sklearn.utils import class_weight,compute_class_weight,shuffle
from keras.activations import relu
from keras import regularizers
from keras.callbacks import EarlyStopping
import random

x = []
y = []
z = []
sm = []

y_test = []
z_test = []
x_test = []

def to_sparse(f_id,z_label):
	row=[]
	col=[]
	val=[]
	row_cnt = 0
	for r in f_id:
		for e in r:
			row.append(row_cnt)
			col.append(e)
			val.append(1)
		row.append(row_cnt)	
		col.append(1999999)
		val.append(int(z_label[row_cnt]))
		row_cnt +=1
	sparse_matrix = csr_matrix((val,(row,col)),shape=(max(row)+1,max(col)+1))
	return sparse_matrix

def get_model():
	input1 = Input(batch_shape =(None,2000000),sparse = True)
	#hidden = Dense(32,activation='relu')(input1)
	out = Dense(2,activation = 'softmax',kernel_initializer='uniform')(input1)#,W_regularizer=regularizers.l2(0.002)
	model = Model(inputs=input1, outputs = out)
	adam = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-08, decay=0.0)
	model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.summary()
	return model


def read_train_data(f,batch_size):

	x=[]
	y=[]
	z=[]
	for i in range(batch_size):
		one_row_data = f.readline()
		one_row_data_arr = one_row_data.split()
		y.append(one_row_data_arr[0])
		z.append(one_row_data_arr[1])
		feature_list=[]
		for j in range(2,len(one_row_data_arr)):
			a = one_row_data_arr[j].split(":")
			feature_id = int(a[0])
			feature_list.append(feature_id)
		x.append(feature_list)
	sm = to_sparse(x,z)
	y_ca = keras.utils.to_categorical(y,2)
	return sm,y_ca

def read_test_data():
	with open('/data2/prune/sp_data/test_out100_0.txt') as ff:
		#print('\nReading testing data from :',address_0)

		#due to unbalanced label of click, we want more label "1"! select them out!
		for i in range(50000):
			one_row_data = ff.readline()
			#one_row_data = line
			one_row_data_arr = one_row_data.split()
			y_test.append(one_row_data_arr[0])
			z_test.append(one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				feature_list.append(feature_id)
			x_test.append(feature_list)
		#Then read another half of data, most of them are "0"!
	with open('/data2/prune/sp_data/test_out100_1.txt') as ff:
		for k in range(50000):
			one_row_data = ff.readline()
			one_row_data_arr = one_row_data.split()
			y_test.append(one_row_data_arr[0])
			z_test.append(one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				feature_list.append(feature_id)
			x_test.append(feature_list)


def mygenerator(batch_size):
	accu=0
	while True:
		with open('/data2/train.yzx.txt') as f:
			if((accu+batch_size)<1000000):
				a,b=read_train_data(f,batch_size)
				accu += batch_size
			else:
				a,b=read_train_data(f,1000000-accu)
				accu = 0
			yield a,b

if __name__ == "__main__":
	read_test_data()
	sm_test = to_sparse(x_test,z_test)
	y_test_ca = keras.utils.to_categorical(y_test,2)

	model = get_model()
	early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
	batch_size = 102400
	history = model.fit_generator(mygenerator(batch_size),steps_per_epoch=int(10000000/batch_size)+1,epochs=20,verbose=1,validation_data=(sm_test,y_test_ca),class_weight={0:1,1:806},callbacks=[early_stopping])
	ynew = model.predict(sm_test)
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(len(y_test)):
		if(ynew[i][0]>ynew[i][1] and y_test[i]=="0"):
			tn +=1
		elif(ynew[i][0]>ynew[i][1] and y_test[i]=="1"):
			fn +=1
		elif(ynew[i][0]<ynew[i][1] and y_test[i]=="0"):
			fp +=1
		elif(ynew[i][0]<ynew[i][1] and y_test[i]=="1"):
			tp +=1
	print("True Negative :",tn)
	print("True Positive :",tp)
	print("False Negative :",fn)
	print("False Positive :",fp)
	#print(shape(sm))
	#print(shape(sm_test))
	#print(d_class_weights)
