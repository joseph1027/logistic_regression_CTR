from scipy.sparse import *
from scipy import *
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Input,Embedding,Dense,Dropout,Activation,Add,concatenate
from keras import optimizers
from sklearn.utils import class_weight,compute_class_weight
from keras.activations import relu
from keras import regularizers
from keras.callbacks import EarlyStopping
import random

#training data size = 363447989
#label_1:450400
#label_0:362997589
#testing data size  = 38980658
#label_1:50594
#label_0:38930064

epoch=1000
#lists for storing the values
y = []#click
z = []#price
x = []#feaure_id
#sm = []

y_test = []
z_test = []
x_test = []
#sm_test= []

def read_train_data(address_0,address_1,data_size):
	with open(address_0) as f:
		print('\nReading training data from :',address_0)
		#due to unbalanced label of click, we want more label "1"! select them out!
		for i in range(int(data_size)*45):
			one_row_data = f.readline()
			#one_row_data = line
			one_row_data_arr = one_row_data.split()
			y.append(one_row_data_arr[0])
			z.append(one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				feature_list.append(feature_id)
			x.append(feature_list)
		#Then read another half of data, most of them are "0"!
	with open(address_1) as g:
		for k in range(int(data_size)*9):
			one_row_data = g.readline()
			one_row_data_arr = one_row_data.split()
			y.append(one_row_data_arr[0])
			z.append(one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				feature_list.append(feature_id)
			x.append(feature_list)

	for i in range(len(y)):
		r = random.randint(0,len(y)-1)
		tmp_x = x[i]
		tmp_y = y[i]
		tmp_z = z[i]
		x[i] = x[r]
		y[i] = y[r]
		z[i] = z[r]
		x[r] = tmp_x
		y[r] = tmp_y
		z[r] = tmp_z 


def read_test_data(address_0,address_1,data_size):
	with open(address_0) as f:
		print('\nReading testing data from :',address_0)

		#due to unbalanced label of click, we want more label "1"! select them out!
		for i in range(int(data_size)):
			one_row_data = f.readline()
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
	with open(address_1) as f:
		for k in range(data_size):
			one_row_data = f.readline()
			one_row_data_arr = one_row_data.split()
			y_test.append(one_row_data_arr[0])
			z_test.append(one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				feature_list.append(feature_id)
			x_test.append(feature_list)

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
	adam = keras.optimizers.Adam(lr=0.00005,beta_1=0.9,beta_2=0.999,epsilon=1e-08, decay=0.0)
	model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.summary()
	return model
	#ynew = model.predict_classes(t_dm[1212])
	#print(ynew)

if __name__ == "__main__":
	data_size = 50000
	read_train_data("/data2/prune/sp_data/train_out20_0.txt","/data2/prune/sp_data/train_out20_1.txt",data_size)
	read_test_data("/data2/prune/sp_data/test_out20_0.txt","/data2/prune/sp_data/test_out20_1.txt",data_size)
	#z = list(map(int, z))
	#z_test = list(map(int, z_test))
	#m_z = max(z)
	#m_z_test = max(z_test)
	#norm_z = [float(q)/m_z for q in z]
	#norm_z_test = [float(p)/m_z_test for p in z_test]
	sm = to_sparse(x,z)
	sm_test = to_sparse(x_test,z_test)
	y_ca = keras.utils.to_categorical(y,2)
	y_test_ca = keras.utils.to_categorical(y_test,2)
	model = get_model()
	y_integers = np.argmax(y_ca, axis=1)
	class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
	d_class_weights = dict(enumerate(class_weights))
	early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
	history = model.fit(sm,y_ca,batch_size=102400,epochs=epoch,verbose=1,validation_data=(sm_test,y_test_ca),class_weight=d_class_weights,callbacks=[early_stopping])#validation_data=(sm_test,y_test_ca)
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
	print(shape(sm))
	print(shape(sm_test))
	print(d_class_weights)
	#print(norm_z)
	