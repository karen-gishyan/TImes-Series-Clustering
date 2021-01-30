import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

from data import *
import numpy as np
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from yellowbrick.cluster import SilhouetteVisualizer


def clustering(data,ncols=15,nclusters=5,preprocess=None, distance_metric="dtw",plot=False,title=None):

	"""
	Performs times series clustering, returns the silhouette score.

	Parameters:
	
	data: pandas DataFrame.
	nclusters: int (default: 5), optional
		Number of clusters to form.

	preprocess: str {"scale_mean_variance","min_max"},  optional
		scale_mean_variance scales times series in each dimension to mean =0.0, std=1.0
		min_max scales the times series between 0.0 and 1.0
	
	distance_metric: str ["dtw","euclidean"], (default: "dtw"), optional
		the distance metric to be used between the series.	
	"""

	ts_list=[]
	names_list=[]
	for label, series in data.iloc[:,:ncols].items():

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
	
	return {"model":km, "silhouette":sil_score,"two_dim_data":two_dim_data}


def visualize_silhoueete(model,data,plot=True):

	vis=SilhouetteVisualizer(model)
	vis.fit(data) # 2D.
	if plot: vis.poof()

