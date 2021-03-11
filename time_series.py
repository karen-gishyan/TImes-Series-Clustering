from data import * 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import time

class TimeSeries:

	def __init__(self,series:pd.Series)-> Optional[str]:

		"""
		Parameters:
		series: array, pandas series object.
		"""
		self.series=series
		self.name=series.name
		

	def decompose(self,**kwargs):

		"""
		Decomposes the times series into seasonal trend residual components.

		Parameters:
			**kwargs: keyword arguments for statsmodels.tsa.seasonal.seasonal_decompose, optional.


		period is inferred from index, alternatively can be set to freq=1, for yearly.
		period =n means we get a cycle after every n observations.
		https://robjhyndman.com/hyndsight/seasonal-periods/

		"""
		self.decomposed_obj=seasonal_decompose(self.series,**kwargs) 
		return self	
	

	def plot_decomposition(self):

		attribute_dict=self.decomposed_obj.__dict__
		number_of_attributes=len(attribute_dict)

		for i, (key, series) in enumerate(attribute_dict.items()):
			
			plt.subplot(number_of_attributes,1,i+1)	
			plt.plot(series,label=key.replace("_","")) # without replace, legends not displayed.
			plt.legend(loc="upper right")
		
		plt.suptitle(self.name)
		plt.subplots_adjust(hspace=0.5)		
		plt.show()

		return self


	def test_stationarity(self,**kwargs)->Tuple[str,pd.Series]:
		
		"""
		Conduct's a Dickey Fuller Test.	
		Parameters
			**kwargs: keyword arguments for statsmodels.tsa.stattools.adfuller,optional 
		"""
		
		self.stationarity_test=adfuller(self.series,**kwargs)		
		output=pd.Series(self.stationarity_test[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])	
		
		for key, value in self.stationarity_test[4].items():
			output["Critical Value {}".format(key)]=value
		
		return (self.name, output)


def decompose_and_test_stationarity(df,ncols:Optional[str]=None,plot:bool=False,**kwargs)->dict:
	
	"""
	Wrapper function for performing decomposition.

	Parameters
		df: pandas DataFrame.
		**kwargs: statsmodels.tsa.seasonal.seasonal_decompose keyword arguments, optional.
		
	"""

	stationarity_dict={}
	for index, column in enumerate(df):

		pts=TimeSeries(df[column])	
		if plot: pts.decompose(**kwargs).plot_decomposition() 
		
		p_value=pts.test_stationarity()[1]["p-value"]
		stationarity_dict["{}. P-value for {} column".format(index, column)]=p_value
		
		if ncols:
			if index==ncols-1: break

	return stationarity_dict

	

if __name__	=="__main__":

	"""
	Sample columns for visualizing decomposition.
	"""
	final_dataset=False
	
	if final_dataset: 
		masters_data=pd.read_csv("datasets/norm_masters.csv")

	else:
		#Preprocessed data before differencing.
		masters_data=pd.read_csv("datasets/masters_before_diff.csv")
	
		masters_data.rename(columns={"Unnamed: 0":'Index'},inplace=True)
		masters_data.set_index("Index",inplace=True)
		masters_data.index=pd.DatetimeIndex(masters_data.index,freq="AS")

		sample_decompose_list=["Number of woman entrants, persons","Number of man entrants, persons",
		"Number of state order training woman entrants, persons","Number of state order training man entrants, persons",
		"Number of state order training woman entrants, persons"]	

	for name in sample_decompose_list:
		ts=masters_data[name]
		
		ts_obj=TimeSeries(ts).decompose(period=2).plot_decomposition() 
		tuple_=ts_obj.test_stationarity(regression='c') # all are stationary becauuse we use the final masters_dataframe.		

		print(tuple_[0])
		print(tuple_[1])
		time.sleep(1)

		### decomposoition period does not affect stationarity results.
		### To check which period is the best for stationairty, residuals can be plotted instead
		### of the whole series (comment out the code bellow.)

		### stationairty results with period two are better.
		# resid_obj=TimeSeries(ts_obj.decomposed_obj.resid.dropna()).test_stationarity()
		# print(resid_obj[0])
		# print(resid_obj[1])
		# time.sleep(1)



