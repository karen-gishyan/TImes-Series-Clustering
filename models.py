import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

from data import *
import numpy as np
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax




def clustering(data,ncols=15,nclusters=5,preprocess=None, distance_metric="dtw"):

	"""
	Performs times series clustering.

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

		ts_list.append(series)
		names_list.append(label)

	ts_data=to_time_series_dataset(np.array(ts_list))
	
	if preprocess=="scale_mean_variance": 
		ts_data=TimeSeriesScalerMeanVariance().fit_transform(ts_data)

	elif preprocess=="min_max":

		ts_data=TimeSeriesScalerMinMax().fit_transform(ts_data)


	#print(ts_data)
	rows=ts_data.shape[1]

	#print(ts_data.shape)
	# print(ts_data)


	km=TimeSeriesKMeans(n_clusters=nclusters,metric="dtw",random_state=11)
	pred=km.fit_predict(ts_data)
	#print(pred)
	# sys.exit()

	plt.figure()
	for cluster in range(nclusters):  # number of clusters
		
		plt.subplot(nclusters, 1, cluster+1)
		

		for i,ts_series in zip(np.argwhere(pred == cluster).ravel(),ts_data[pred == cluster]): # which series belongs to which cluster.
			
			plt.plot(ts_series.ravel(), "k-",alpha=0.5,label=names_list[i])

		plt.plot(km.cluster_centers_[cluster].ravel(), "b")
		plt.legend(loc="upper right")   
		# plt.xlim(0, rows)
		
		plt.title("Cluster %d" % (cluster + 1))

	plt.tight_layout()

	plt.show()

	sil_score=silhouette_score(ts_list, pred, metric="dtw")
	print("The silhouette_score is",sil_score) # understand what this means.



if __name__ == '__main__':

	#if decomposition needed, should be decomposed, appended to a new dataset, then clustered.
	
	#expected procedure.
	masters_data=read_csv("datasets/masters.csv",save=True,save_title="masters_final")
	clustering(masters_data,preprocess="min_max")


