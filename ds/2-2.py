import csv

with open('iris.data', 'r') as data_stream:
	for n, row in enumerate(csv.DictReader(data_stream, 
			fieldnames=['sepal_length', 'sepal_height', 'petal_length', 'petal_height', 'target'],
		 	dialect='excel')):
		if n == 0:
			print(n, row)
		else:
			break

