from data import *
from statsmodels.tsa.seasonal import seasonal_decompose
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

		self.decomposed_obj=seasonal_decompose(self.series,model=method) #statsmodels class.
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
		
		self.stationarity_test=adfuller(self.series,autolag=auto_lag,maxlag=max_lag,regression=regression)
		return self.stationarity_test



if __name__ == '__main__':

	#expected procedure.
	phd_data=read_csv("datasets/phd.csv",save=True,save_title="phd_final")
	ts=TimesSeries(phd_data.iloc[:,50]).decompose().plot_decomposition() # method cascading through return self.
	res=ts.test_stationarity()
	print(res)