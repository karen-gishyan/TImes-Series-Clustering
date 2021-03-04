import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import os 
import sys


def read_csv(path,save=False,save_title="output"):
	data=pd.read_csv(path,sep=";",skiprows=2) 
	
	data=data.set_index(data.columns[0]) # the variable names as indices.
	data=data.transpose() # transposes the indexes and columns.
	
	
	data.index=pd.DatetimeIndex(data.index,freq="AS")

	### seting index manually. much more robust than the commented code above, which infers freq and only allows AS.
	### however, as we deal with two datasets, DatetimeIndex is more preferable.
	
	#data.index=pd.date_range(start="2001-01-01",periods=19,freq="AS") 

	if save:
		path=os.path.join(os.getcwd(),"datasets",save_title+".csv") 
		
		if not os.path.exists(path): data.to_csv(path)
	return data


def visualize_series(data, column_list=None,ncols=3):
	
	"""
	Plots the dataset columns.

	Parameters:
	
	column_list: list, optional
		list of column names to visualize.

	ncols: int (default=3) 
		Number of columns to visualize if no column list is specified.
	"""

	for i, column in enumerate(data.columns):

		if column_list:
			if column in column_list:			
				plt.plot(data[column],label=column)
				plt.legend()
				plt.show()

		else:
			plt.plot(data[column],label=column)
			plt.legend()
			if i==ncols:	
				break			
		plt.show()

	
	