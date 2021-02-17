import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from data import *
import numpy as np
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from yellowbrick.cluster import SilhouetteVisualizer
from times_series import *


def clustering(data,ncols=None,nclusters=5,preprocess=None, distance_metric="dtw",plot=False,title=None):

	"""
	Performs times series clustering, returns the silhouette score.

	Parameters:
	
	data: pandas DataFrame.
	ncols: int (default None)
		Number of dataset columns to subset from o index.
	nclusters: int (default: 5), optional
		Number of clusters to form.

	preprocess: str {"scale_mean_variance","min_max"},  optional
		scale_mean_variance scales times series in each dimension to mean =0.0, std=1.0
		min_max scales the times series between 0.0 and 1.0
	
	distance_metric: str ["dtw","euclidean"], (default: "dtw"), optional
		the distance metric to be used between the series.	
	"""


	ts_list,names_list =[],[]
	number_of_cols=data.shape[1]

	if not ncols: ncols= number_of_cols

	else:
		assert ncols< number_of_cols 


	for label, series in data.iloc[:,:ncols].items(): # if ncols >number_of_cols pandas handles  it by taking max cols, but we make an assertion.

		ts_list.append(series) # series become rows.
		names_list.append(label)

	two_dim_data=np.array(ts_list) 
	#print(np.array(ts_list).shape)	
	ts_data=to_time_series_dataset(two_dim_data)
	
	if preprocess=="scale_mean_variance": 
		ts_data=TimeSeriesScalerMeanVariance().fit_transform(ts_data)

	elif preprocess=="min_max":

		ts_data=TimeSeriesScalerMinMax().fit_transform(ts_data)
	
	rows=ts_data.shape[1]
	km=TimeSeriesKMeans(n_clusters=nclusters,metric="dtw",random_state=11)
	pred=km.fit_predict(ts_data)
	
	if plot:

		plt.figure()

		for cluster in range(nclusters):  
			
			plt.subplot(nclusters, 1, cluster+1)
			
			for i,ts_series in zip(np.argwhere(pred == cluster).ravel(),ts_data[pred == cluster]): # which series belongs to which cluster.
				
				plt.plot(ts_series.ravel(),"k-", alpha=0.3,label=names_list[i]) 
				#plt.plot(ts_series.ravel(),"k-", alpha=0.3) 


			plt.plot(km.cluster_centers_[cluster].ravel(), "b")
			plt.legend(loc="upper left",bbox_to_anchor=(1,1))  
			# plt.xlim(0, rows)
			
			plt.title("Cluster %d" % (cluster + 1))

		if title: plt.suptitle("{}".format(title))
		
		plt.tight_layout()
		plt.show()

	sil_score=silhouette_score(ts_list, pred, metric="dtw")	 

	
	return {"model":km, "silhouette":sil_score,"two_dim_data":two_dim_data,"n_cols":ncols,"n_clusters":nclusters}


def visualize_silhoueete(model,data,plot=True):

	vis=SilhouetteVisualizer(model)
	

	vis.fit(data) # 2D.
	if plot: vis.poof()



def decompose_and_test_stationarity(df,ncols=None,plot=False):
	
	"""
	Returning the P values of stationary for each columns.
	P-value of more than 5% indicates non-stationarity,
	"""

	stationarity_dict={}
	for index, column in enumerate(df):

		pts=TimesSeries(df[column])	
		if plot: pts.decompose(method="additive").plot_decomposition()
		
		# print(column)
		# print(pts.test_stationarity())
		# print("-------")

		p_value=pts.test_stationarity()["p-value"] # Only for p values.
		stationarity_dict["{}. P-value for {} column".format(index, column)]=p_value
		
		if ncols:
			if index==ncols-1: break

	return stationarity_dict