from urllib.request import urlopen
import pandas as pd 

URL = 'http://aima.cs.berkeley.edu/data/iris.csv'
iris_p = urlopen(URL)
iris_other = pd.read_csv(iris_p, sep = ',', decimal='.', header  = None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
iris_other.head()
