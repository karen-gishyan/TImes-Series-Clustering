from tslearn.utils import to_time_series,to_time_series_dataset
import matplotlib.pyplot as plt 

def tutorial_function():

	"""
	Basic intro for the tslearn's series and dataset construction. 
	"""

	series=[1,3,4,2]
	formated_tseries=to_time_series(series) # 4 rows and 1 column.
	print("The shape and type of the series are {}, {}".format(formated_tseries.shape,type(formated_tseries)))
	#print(formated_tseries)

	second_series=[1,2,4,2,5]
	formated_dataset=to_time_series_dataset([series,second_series])
	print("The shape of the dataset is {}".format(formated_dataset.shape)) # 3 dimensional-  2 ts datasets, with 4 rows and 1 column.
	#print(formated_dataset) 


