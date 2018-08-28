cities=[]
f=open('cities.txt')
for line in f:
	cities.append(line.strip().split(','))
f.close()

f=open('cities2.txt','w')
for city in cities:
	f.write('%d,%d,%d\n' % (int(city[0]),round(float(city[2])),0 if city[3]=='중부' else 1))
f.close()