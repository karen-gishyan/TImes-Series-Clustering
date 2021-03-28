###  Main procedure.
###  Check decomposition, stationarity. -✓
###  Check decompostion, stationarity relation, do series transformation if needed. -✓ 
###  Check dim reduction.-✓
###  Check standardization.-✓
###  Check iteratively selecting ks. -✓
###  Evaluete using silhouette scores, and visualize. -✓

from cluster import *
from data import *
import seaborn as sns
from time_series import test_stationarity, TimeSeries
from sklearn.preprocessing import MinMaxScaler
import re
import time


masters_data=read_csv("datasets/masters_original.csv")
phd_data=read_csv("datasets/phd_original.csv")

### removes trailing space before the comma.
masters_data.columns=[re.sub('\\s*([,])s*', r'\1', mcol) for mcol in masters_data.columns]
phd_data.columns=[re.sub('\\s*([,])s*', r'\1', pcol) for pcol in phd_data.columns]

### removes trailing multiple spaces in the string.
masters_data.columns=[re.sub(' +', ' ', mcol) for mcol in masters_data.columns]
phd_data.columns=[re.sub(' +', ' ', pcol) for pcol in phd_data.columns]


### list of needed master's columns, most of which exist in phd as well.
### We make sure all women are converted to woman and men to man in col names
### to later compare master's and phd dataset col names..

master_cols=["Number of women entrants, persons","Number of men entrants, persons",
"Number of state order training women entrants, persons","Number of state order training men entrants, persons",
"Number of paid training women entrants, persons","Number of paid training men entrants, persons",
"Number of woman students, persons","Number of man students, persons","Number of state order training woman students, persons",
"Number of state order training man students, persons","Number of paid training woman students, persons",
"Number of paid training man students, persons","Number of woman graduates, persons","Number of man graduates, persons",
"Number of foreign woman entrants, persons","Number of foreign man entrants, persons","Number of foreign woman students, persons","Number of foreign man students, persons",
"Number of state order training foreign woman students, persons","Number of state order training foreign man students, persons",
"Number of paid training foreign woman students, persons","Number of paid training foreign man students, persons"]

### decompose the columns names to words, change men -> man, women -> woman, rejoing into strings, rejoin into a list.
m_cols=[]
for string in master_cols:

	new_col_name,renamed_list_of_strings=" ",[]
	for split_string in string.split():

		if split_string=="women": split_string="woman"			
		elif split_string=="men": split_string="man"
			
		renamed_list_of_strings.append(split_string)	
	m_cols.append(new_col_name.join(renamed_list_of_strings)) 

master_limited,phd_limited=pd.DataFrame(),pd.DataFrame()

### new dataframe with men -> man and women -> woman in col_names.
for  new_col,old_col in zip(m_cols,master_cols): 
	master_limited[new_col]=masters_data[old_col]


### finds the columns in the masters_data which are not present in the phd data.
diff_list=[i for i in master_limited.columns if i not in phd_data.columns]

master_limited.drop(diff_list,axis=1,inplace=True)

phd_cols=master_limited.columns


for pcol in phd_cols:
	phd_limited[pcol]=phd_data[pcol]


assert len(master_limited.columns)==len(phd_limited.columns)


print("Stationarity test for the Masters Dataset")

#print(test_stationarity(master_limited))

print("Stationarity test for the PhD Dataset")
#print(test_stationarity(phd_limited))



### Differencing the series to period 1 to remove stationarity.
masters_diff=master_limited.diff().dropna()
phd_diff=phd_limited.diff().dropna()


print("Number of rows in Masters before and after idfferencing is {} and {}.".format(master_limited.shape[0],masters_diff.shape[0]))
print("Number of rows in PhD before and after differencing is {} and {}.".format(phd_limited.shape[0],phd_diff.shape[0]))
print("---")


#print("Stationarity test for the Masters Dataset after Differencing")
first_order_stationary_masters=test_stationarity(masters_diff)
print(first_order_stationary_masters)

print("Stationarity test for the PhD Dataset after Differencing")
first_order_stationary_phd=test_stationarity(phd_diff)
print(first_order_stationary_phd)
sys.exit()

### Finding non-stationary columns after 2nd differencing.

#print("Master's DataSet subset after 2nd differencing")
subsetm=masters_diff.iloc[:,non_stationary_masters_list]
subsetm=subsetm.diff().dropna() # 2002 year  is dropped.

second_diff_masters_results=decompose_and_test_stationarity(subsetm) # only 1 column is not stationary after 2nd differencing.
non_stationary_masters_second_diff_list=[i for i, value in enumerate(second_diff_masters_results.values()) if value >0.05]


cols=list(subsetm.iloc[:,non_stationary_masters_second_diff_list].columns)

master_limited.drop(cols,axis=1,inplace=True)

### The column which renamed non-stationry after second differencing was dropped.

#print(master_limited.columns)
subsetm.drop(subsetm.columns[non_stationary_masters_second_diff_list],axis=1,inplace=True)

masters_diff.drop(masters_diff.columns[non_stationary_masters_list],axis=1,inplace=True)
masters_diff=masters_diff.merge(subsetm,right_index=True,left_index=True) # merging stationary columns.

#print("PhD's DataSet subset after 2nd differencing")
subsetp=phd_diff.iloc[:,non_stationary_phd_list]
subsetp=subsetp.diff().dropna()

second_diff_phd_results=decompose_and_test_stationarity(subsetp) 
non_stationary_phd_second_diff_list=[i for i, value in enumerate(second_diff_phd_results.values()) if value >0.05]


### removing non stationary columns from the original phd dataset.
cols=list(subsetp.iloc[:,non_stationary_phd_second_diff_list].columns)
phd_limited.drop(cols,axis=1,inplace=True)

# The columns which remain non-stationary after second differencing are removed.
subsetp.drop(subsetp.columns[non_stationary_phd_second_diff_list],axis=1,inplace=True)

phd_diff.drop(phd_diff.columns[non_stationary_phd_list],axis=1,inplace=True)
phd_diff=phd_diff.merge(subsetp,right_index=True,left_index=True) # merging stationary- columns.


print("Final Masters and PhD  dataset shapes are {} and {}.".format(masters_diff.shape,phd_diff.shape))

### Column wise min-max normalization from 0 to 1.
### The decomposition results stay the same after normalization.
### Normalization does not change stationarity, so 2nd differencing was needed.

### if true, removes non-stationary columns from the original.
### aims to see if differencing helps in increaseing silhouette score, but it does not.


# sns.heatmap(master_limited.corr(),annot=True)
# plt.show()
# sns.heatmap(masters_diff.corr(),annot=True)
# plt.show()

undifferenced_dataasets=False #default differenced.
if undifferenced_dataasets:
	
	assert len(master_limited.columns)==len(masters_diff.columns)
	assert len(phd_limited.columns)==len(phd_diff.columns)
	masters_diff=master_limited
	phd_diff=phd_limited


### STL Decomposition.
# stl_decompose=False
# seasonal_decompose=False

# stl_decomposed_masters_df={}
# for name, series in master_limited.iteritems():

# 	ts_obj=TimeSeries(series).decompose(type='stl',period=7)
# 	res=ts_obj.decomposed_obj.resid.dropna()
# 	t=TimeSeries(res).test_stationarity()
# 	# print(t[0])
# 	# print(t[1])
# 	# time.sleep(1)
# 	stl_decomposed_masters_df.update({name:res})



# stl_decomposed_masters_df=pd.DataFrame.from_dict(stl_decomposed_masters_df)

# stl_decomposed_phd_df={}

# for name, series in phd_limited.iteritems():

# 	ts_obj=TimeSeries(series).decompose(type='stl',period=2)
# 	res=ts_obj.decomposed_obj.resid.dropna()
# 	stl_decomposed_phd_df.update({name:res})

# stl_decomposed_phd_df=pd.DataFrame.from_dict(stl_decomposed_phd_df)

# ### Seasonal Decomposition.
# seasonal_decomposed_masters_df={}
# for name, series in master_limited.iteritems():

# 	ts_obj=TimeSeries(series).decompose(period=2)
# 	res=ts_obj.decomposed_obj.resid.dropna()
# 	seasonal_decomposed_masters_df.update({name:res})

# seasonal_decomposed_masters_df=pd.DataFrame.from_dict(seasonal_decomposed_masters_df)

# seasonal_decomposed_phd_df={}

# for name, series in phd_limited.iteritems():

# 	ts_obj=TimeSeries(series).decompose(period=2)
# 	res=ts_obj.decomposed_obj.resid.dropna()
# 	t=TimeSeries(res).test_stationarity()
# 	#print(t[0]) 
# 	#print(t[1])
# 	#time.sleep(1)
# 	seasonal_decomposed_phd_df.update({name:res})

# seasonal_decomposed_phd_df=pd.DataFrame.from_dict(seasonal_decomposed_phd_df)

# ###

# if seasonal_decompose:
# 	masters_diff=seasonal_decomposed_masters_df
# 	phd_diff=seasonal_decomposed_phd_df
# 	print("1")

# if stl_decompose:

# 	masters_diff=stl_decomposed_masters_df
# 	phd_diff=stl_decomposed_phd_df
# 	print("1")

normalized_masters_diff=(masters_diff-masters_diff.min())/(masters_diff.max()-masters_diff.min())
#print(decompose_and_test_stationarity(normalized_masters_diff))

normalized_phd_diff=(phd_diff-phd_diff.min())/(phd_diff.max()-phd_diff.min())

#print(decompose_and_test_stationarity(normalized_phd_diff))

print("Final Masters and PhD shapes in the normalized datasets are {} and {}.".format(normalized_masters_diff.shape,normalized_phd_diff.shape))
print("---")
### In the clustering, these normalized masters and phd datasets are again transposed.
### Determining the best k for clustering.
### With preprocessing, the silhoueete scores decrease.

	
k=10
print("Masters data clustering results.")
print("---")

for cluster in range(2,k+1):
	
	### decorator applied with the first approach 
	### (without passing to clustering decorator manually)
	res=clustering(normalized_masters_diff,nclusters=cluster,distance_metric="softdtw",pca=True) # preprocess="min_max"
	print("The silhouette score for {} clusters is {}.".format(cluster,round(res["silhouette"],3)))


# print("---")
print("PhD data clustering results.")
print("---")


for cluster in range(2,k+1):	
	
	res=clustering(normalized_phd_diff,nclusters=cluster,distance_metric="softdtw",pca=True) 
	print("The silhouette score for {} clusters is {}.".format(cluster,round(res["silhouette"],3)))

time.sleep(5)


### Optimal clusters with silhouette scores with normalized datasets.
vis=True #for silhouette visualization.

print("---")

### decorator call with arguments, for both master's and phd,second approach.
### Metrics do not need to be the same, but the same metric in genereal produces higher results.
res=clustering_decorator(vis,visualize_silhoueete,
	distance_metric="softdtw")(clustering)(normalized_masters_diff,
	nclusters=2,plot=True,distance_metric="softdtw",title="Adjusted Time Series: Master's Dataset",pca=True)


rd=res["dict_of_cluster_names"]


for cluster,columns in rd.items():
	print("For Masters :", cluster,"has {} series.".format(len(columns)))

print("---")

res=clustering_decorator(vis,visualize_silhoueete,distance_metric="softdtw")(clustering)(normalized_phd_diff,nclusters=3,
	plot=True,distance_metric="softdtw",title="Adjusted Time Series: PhD Dataset",pca=False)

rd=res["dict_of_cluster_names"]

for cluster,columns in rd.items():
	print("For PhD :", cluster,"has {} series.".format(len(columns)))	

