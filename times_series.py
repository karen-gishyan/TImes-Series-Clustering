from data import *
from utils import *
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller


class TimesSeries:

	def __init__(self,series):

		"""
		Parameters:
		series: array, pandas series object.
		"""
		self.series=series
		self.name=series.name
		

	def decompose(self,method="additive"):

		"""
		Decomposes the times series into seasonal trend residual components.

		Parameters:

		method: str {'additive','multiplicative'}, (default additive), optional
			decompostion method.
		"""

		### period is inferred from index, alternatively can be set to freq=1, for yearly.
		### period =n means we get a cycle after every n observations.
		### https://robjhyndman.com/hyndsight/seasonal-periods/
		self.decomposed_obj=seasonal_decompose(self.series,model=method,period=2) #statsmodels.tsa.seasonal.DecomposeResult class.
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


	def test_stationarity(self,max_lag=None,regression='c',auto_lag='AIC'):
		
		"""
		Conduct's a Dickey Fuller Test.
		"""
		
		self.stationarity_test=adfuller(self.series,autolag=auto_lag,maxlag=max_lag,regression=regression)
		
		output=pd.Series(self.stationarity_test[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])	
		
		for key, value in self.stationarity_test[4].items():
			output["Critical Value {}".format(key)]=value
		
		return output


	 
if __name__	=="__main__":
	
	masters_data=read_csv("datasets/masters.csv")
	
	#decompose_and_test_stationarity(masters_data,ncols=1,plot=True)
	ts=TimesSeries(masters_data.iloc[:,7])
	#print(ts.test_stationarity())

	### with periods= 2, our residuals demonstrate the best stationarity.
	sample=ts.decompose()	
	residuals=sample.decomposed_obj.resid.dropna()
	print(TimesSeries(residuals).test_stationarity()) 	
	#TimesSeries(residuals).decompose().plot_decomposition()s