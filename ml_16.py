# ml_16.py
# one-hot-encoding

data=[[25,172,'서울',1],[53,169,'부산',0],[49,177,'대구',1]]

loc={'서울':[1,0,0], '부산':[0,1,0], '대구':[0,0,1]}

onehot=[]
for row in data:
	l=[ row[0], row[1] ]
	l=l+loc[row[2]]
	l.append(row[3])
	onehot.append(l)
	
print(onehot)
