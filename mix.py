with open('/data2/train_label_0.txt') as f:
	with open('/data2/train_label_1.txt') as g:
		with open('/data2/train_mix_1_1_300000.txt','a') as w:
			for i in range(300000):
				g_line = g.readline()
				f_line1 = f.readline()
				#f_line2 = f.readline()
				#f_line3 = f.readline()
				w.write(g_line)
				w.write(f_line1)
				#w.write(f_line2)
				#w.write(f_line3)
