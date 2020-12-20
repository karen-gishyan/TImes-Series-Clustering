import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

from data import *
import numpy as np
from tslearn.utils import to_time_series,to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance




def tutorial_function():


	series=[1,3,4,2]
	formated_tseries=to_time_series(series) # 4 rows and 1 column.
	print("The shape and type of the series are {}, {}".format(formated_tseries.shape,type(formated_tseries)))
	#print(formated_tseries)

	second_series=[1,2,4,2,5]
	formated_dataset=to_time_series_dataset([series,second_series])
	print("The shape of the dataset is {}".format(formated_dataset.shape)) # 3 dimensional-  2 ts datasets(can consider series), with 4 rows and 1 column.
	print(formated_dataset) 

#tutorial_function()



masters_data=read_csv("datasets/masters.csv",save=True,save_title="masters_final")


ts_list=[]
for label, series in masters_data.iloc[:,:15].items():

	ts_list.append(series)


ts_data=to_time_series_dataset(np.array(ts_list))

rows=ts_data.shape[1]

print(ts_data.shape)
#sys.exit()


km=TimeSeriesKMeans(n_clusters=5,metric="dtw")
pred=km.fit_predict(ts_data)


plt.figure()
for cluster in range(3):  # number of clusters
    
    plt.subplot(3, 1, cluster+1)
    
    for ts_series in ts_data[pred == cluster]:
        plt.plot(ts_series.ravel(), "k-",alpha=0.5)
    plt.plot(km.cluster_centers_[cluster].ravel(), "b")
    #plt.xlim(0, rows)
    #plt.ylim(-4, 4)
    plt.title("Cluster %d" % (cluster + 1))

plt.tight_layout()
plt.show()



# to do.
#standardization, title selection. # check if decompostion needs to be done.



