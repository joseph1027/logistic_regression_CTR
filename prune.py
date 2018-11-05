count = [0 for i in range(2000000)]
x = []
y = []
z = []

with open('/data2/prune/train.txt') as f:
	for line in f:
		one_row_data_arr = line.split()
		for j in range(2,len(one_row_data_arr)):
			a = one_row_data_arr[j].split(":")
			feature_id = int(a[0])
			count[feature_id] +=1
	for i in range(2000000):
		if(count[i]!=0):
			print(i,' : ',count[i])
	print('----------------------------------')

with open('/data2/prune/test.txt') as k:
	for line in k:
		one_row_data_arr = line.split()
		for j in range(2,len(one_row_data_arr)):
			a = one_row_data_arr[j].split(":")
			feature_id = int(a[0])
			count[feature_id] +=1
	for i in range(2000000):
		if(count[i]!=0):
			print(i,' : ',count[i])


with open('/data2/prune/train.txt') as f:
	with open('/data2/prune/train_out100.txt','a') as w:
		for line in f:
			one_row_data_arr = line.split()
			w.write(one_row_data_arr[0]+' '+one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				if(count[feature_id]>=100):
					w.write(' '+str(feature_id)+':1')
				#count[feature_id] +=1
			w.write('\n')

with open('/data2/prune/test.txt') as f:
	with open('/data2/prune/test_out100.txt','a') as w:
		for line in f:
			one_row_data_arr = line.split()
			w.write(one_row_data_arr[0]+' '+one_row_data_arr[1])
			feature_list=[]
			for j in range(2,len(one_row_data_arr)):
				a = one_row_data_arr[j].split(":")
				feature_id = int(a[0])
				if(count[feature_id]>=100):
					w.write(' '+str(feature_id)+':1')
				#count[feature_id] +=1
			w.write('\n')
