import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
from data import *
import numpy as np
import statistics
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.metrics import cdist_dtw, cdist_soft_dtw_normalized
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.clustering import silhouette_score as tslearn_silhouette
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score as sklearn_silhouette, silhouette_samples
from collections import defaultdict
from typing import Optional, Dict, List,Callable


def clustering_decorator(visualize: bool =False,
	silhouette_visualizer: Optional[str]=None,**silkwargs) -> Callable:	
	
	"""
	Clustering decorator for the clustering function. Checks for assertion and 
	checks for Silhouette visualization.
	"""
	
	def function_taker(function: Callable) -> Callable:

		def wrapper(*args,**kwargs)-> Dict:
			"""
			Parameters
				  *args:  arguments of the clustering() function.
				**kwargs: keyword arguments of the clustering() function, optional.
			"""
			result=function(*args,**kwargs)		
			assert result["n_cols"]-1>result["n_clusters"], "Number of columns should be at least by 2 more than the maximum cluster number."				
			
			if visualize:
				silhouette_visualizer(result["model"],result["two_dim_data"],**silkwargs)

			return result
		return wrapper

	return function_taker


@clustering_decorator()

def clustering(data: pd.DataFrame,ncols: Optional[str]=None, nclusters: int =5,preprocess: Optional[str]=None, 
	distance_metric: str="dtw",plot: bool=False,title: Optional[str]=None) -> Dict:

	"""
	Performs times series clustering, 
	returns a dictionary containing the model, silhouette score, data in 2D,
	number of columns, number of clusters, and dictionary of lists of names for each cluster.
	"""

	ts_list,names_list =[],[]
	number_of_cols=data.shape[1]

	if not ncols: ncols= number_of_cols
	else: assert ncols< number_of_cols
		 
	for label, series in data.iloc[:,:ncols].items(): # if ncols >number_of_cols pandas handles it by taking max cols.

		ts_list.append(series) 
		names_list.append(label)

	two_dim_data=np.array(ts_list) # columns to rows (samples for clustering)
	#print(np.array(ts_list).shape)	
	ts_data=to_time_series_dataset(two_dim_data)
	
	if preprocess=="scale_mean_variance": 
		ts_data=TimeSeriesScalerMeanVariance().fit_transform(ts_data)

	elif preprocess=="min_max":
		ts_data=TimeSeriesScalerMinMax().fit_transform(ts_data)
	
	km=TimeSeriesKMeans(n_clusters=nclusters,metric=distance_metric,random_state=11) 
	pred=km.fit_predict(ts_data) # predicts cluster index for each sample.

	dict_of_dicts=defaultdict(dict) # dict equivalent to lambda:{}
	dict_of_lists=defaultdict(list) # list equivalent to lambda:[]
	
	if plot:

		fig=plt.figure(figsize=(7,7))

		### shared x and y titles.
		fig.text(0.5,0.021, "Time", ha="center", va="center")
		fig.text(0.05,0.5, "Values", ha="center", va="center", rotation=90)
		
		for cluster in range(nclusters):  
			
			plt.subplot(nclusters, 1, cluster+1)	

			for i,ts_series in zip(np.argwhere(pred == cluster).ravel(),ts_data[pred == cluster]): # which series belongs to which cluster.				
				
				series=ts_series.ravel()
				series_label=names_list[i]
					
				dict_of_lists[f" Cluster {cluster+1}"].append(series_label)
				#dict_of_dicts[f" The columns and values for cluster {cluster+1} are"].update({series_label:series})
				
				plt.plot(data.index.to_pydatetime(),series,"k-", alpha=0.5,label=series_label) #,label=series_label  
					

			plt.plot(data.index.to_pydatetime(),km.cluster_centers_[cluster].ravel(), "b")
			
			### obtained to change the xaxis frequency.
			ax=plt.gca()
			ax.xaxis.set_major_locator(mdates.YearLocator(base=1)) 
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
			plt.gcf().autofmt_xdate()# creates x axes seperately and rotates.
			
			plt.legend(loc="upper left",bbox_to_anchor=(1,1))  
			plt.title("Cluster %d" % (cluster + 1))

		if title: plt.suptitle("{}".format(title))
				 
		#plt.tight_layout()
		fig.subplots_adjust(left=0.1, right=0.5, bottom=0.08, hspace=0.23)
		plt.show()

	#sil_score=sklearn_silhouette(two_dim_data, pred, metric="euclidean") #jaccard.	 
	sil_score=tslearn_silhouette(two_dim_data, pred, metric=distance_metric) 
	return {"model":km, "silhouette":sil_score,"two_dim_data":two_dim_data,"n_cols":ncols,"n_clusters":nclusters,"dict_of_cluster_names":dict_of_lists}


class Visualize_Silhouette(SilhouetteVisualizer):

	"""
	Inherits from SilhouetteVisualizer and
	modifies the fit method to include dtw and softdtw implementations.
	"""

	def __init__(self,estimator: TimeSeriesKMeans,data:pd.DataFrame,distance_metric: str="dtw"):

		"""
		estimator is a TimeSeriesKMeans or equivalently an a Scikit-Learn clusterer instance.
		"""
		self.distance_metric=distance_metric
		self.data=data
		super().__init__(estimator) 

	def fit(self):

		self.n_samples_=self.data.shape[0]
		self.n_clusters_=self.estimator.n_clusters
		labels=self.estimator.fit_predict(self.data)
		
		self.silhouette_score_ = tslearn_silhouette(self.data, labels,metric=self.distance_metric)
		
		if self.distance_metric=="dtw":
			self.silhouette_samples_ = silhouette_samples(cdist_dtw(self.data), labels,metric="precomputed") # dtw matrix is passed as an argument.	

		elif self.distance_metric=="softdtw":
			self.silhouette_samples_ = silhouette_samples(cdist_soft_dtw_normalized(self.data), labels,metric="precomputed")
		
		else:
			self.silhouette_samples_ = silhouette_samples(self.data, labels,metric=self.distance_metric)

		self.draw(labels)	
		return self


def visualize_silhoueete(model:TimeSeriesKMeans,data:pd.DataFrame,distance_metric: str="dtw",plot: bool=True)-> Optional[str]:

	"""
	wrapper function for instantiating Visualize_Silhouette .
	"""
	vis=Visualize_Silhouette(model,data,distance_metric) 
	vis.fit() # 2D.
	if plot: vis.poof()


