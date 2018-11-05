with open('/data2/prune/train_out20.txt')as f:
	with open('/data2/prune/sp_data/train_out20_0.txt','a')as a:
		with open('/data2/prune/sp_data/train_out20_1.txt','a')as b:
			for line in f:
				one_row_data_arr = line.split()
				if(one_row_data_arr[0]=='0'):
					a.write(line)
				elif(one_row_data_arr[0]=='1'):
					b.write(line)
