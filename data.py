import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import sys


def read_csv(path,save=False,save_title="output"):
	
	data=pd.read_csv(path,sep=";",skiprows=2) 
	data=data.set_index(data.columns[0]) # sets the row label.
	data=data.T 
	

	if save:
		path=os.path.join(os.getcwd(),"datasets",save_title+".csv") 
		
		if not os.path.exists(path): data.to_csv(path)
	return data


def visualize_series(data, column_list=None,ncols=3):
	
	"""
	column_list: list of column names to visualize.
	ncols: Number of columns to visualize if no column list is specified.
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



if __name__=="__main__":
	
	masters_data=read_csv("datasets/masters.csv",save=True,save_title="masters_final")
	phd_data=read_csv("datasets/phd.csv",save=True,save_title="phd_final")
	

	visualize_series(data=masters_data)
	print(masters_data.shape)
	print(phd_data.columns)
	