import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import os 
import sys
from typing import Optional, Dict, List,Callable, Tuple


def read_csv(path: str,save: bool =False,save_title: str="output")-> Optional[str]:
	
	data=pd.read_csv(path,sep=";",skiprows=2) 
	data=data.set_index(data.columns[0]) # the variable names as indices.
	data=data.transpose() 
	
	data.index=pd.DatetimeIndex(data.index,freq="AS")
	#data.index=pd.date_range(start="2001-01-01",periods=19,freq="AS") 
	if save:
		path=os.path.join(os.getcwd(),"datasets",save_title+".csv") 
		
		if not os.path.exists(path): data.to_csv(path)
	
	return data
