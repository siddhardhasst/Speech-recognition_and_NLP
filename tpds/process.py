data_path = 'movie_lines.txt'
with open(data_path, 'r') as f: 
	lines = f.read().split('\n')
op = open ('data.txt', 'w')
op2 = open('data1.txt','w')
i = 0
while i < len(lines):
	print lines[i].split('+++$+++')[4].split('\t')
	try:
		line = []
		line = lines[i].split('+++$+++')[4].split('\t')
		st = ""
		for j in range(0,len(line)):
			st = st+line[j]+" "
		line1 = []
		line1 = lines[i+1].split('+++$+++')[4].split('\t')
		st1 = ""
		for j in range(0,len(line1)):
			st1 = st1+line1[j]+" "
		op.write(st+'\t'+st1+'\n')
		#op1.write(lines[i].split('+++$+++')[4].split('\t')[0] + '\t'+lines[i+1].split('+++$+++')[4].split('\t')[0]+'\n' )
	except:
		break
	i += 2
op.close()