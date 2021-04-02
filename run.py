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


# master_limited.corr().to_csv("correlation_datasets/master_limited.csv")
# phd_limited.corr().to_csv("correlation_datasets/phd.csv")


assert len(master_limited.columns)==len(phd_limited.columns)

print("Stationarity test for the Masters Dataset")

#print(test_stationarity(master_limited))

print("Stationarity test for the PhD Dataset")
#print(test_stationarity(phd_limited))


### Differencing the series to period 1 to remove stationarity.
masters_first_diff=master_limited.diff().dropna()
phd_first_diff=phd_limited.diff().dropna()

# masters_first_diff.corr().to_csv("correlation_datasets/masters_first_diff.csv")
# phd_first_diff.corr().to_csv("correlation_datasets/phd_first_diff.csv")
# sys.exit()

print("Number of rows in Masters before and after idfferencing is {} and {}.".format(master_limited.shape[0],masters_first_diff.shape[0]))
print("Number of rows in PhD before and after differencing is {} and {}.".format(phd_limited.shape[0],phd_first_diff.shape[0]))
print("---")


#print("Stationarity test for the Masters Dataset after Differencing")
first_order_stationary_masters=test_stationarity(masters_first_diff)

#print("Stationarity test for the PhD Dataset after Differencing")
first_order_stationary_phd=test_stationarity(phd_first_diff)

### Finding non-stationary columns after 2nd differencing.

drop_columns=["Number of state order training man entrants, persons","Number of paid training man entrants, persons",
"Number of woman students, persons","Number of state order training man students, persons",'Number of paid training woman students, persons',
'Number of woman graduates, persons','Number of man graduates, persons','Number of foreign man entrants, persons',
'Number of foreign woman students, persons']

drop=True

if drop:

	masters_first_diff.drop(drop_columns,axis=1,inplace=True)
	phd_first_diff.drop(drop_columns,axis=1,inplace=True)
	assert len(masters_first_diff.columns)==len(phd_first_diff.columns)


print("Final Masters and PhD  dataset shapes are {} and {}.".format(masters_first_diff.shape,phd_first_diff.shape))

normalized_master_limited=(master_limited-master_limited.min())/(master_limited.max()-master_limited.min())
normalized_masters_first_diff=(masters_first_diff-masters_first_diff.min())/(masters_first_diff.max()-masters_first_diff.min())
#normalized_masters_sklearn=MinMaxScaler().fit_transform(masters_first_diff) # 
#print(decompose_and_test_stationarity(normalized_masters_diff))

normalized_phd_limited=(phd_limited-phd_limited.min())/(phd_limited.max()-phd_limited.min())
normalized_phd_first_diff=(phd_first_diff-phd_first_diff.min())/(phd_first_diff.max()-phd_first_diff.min())


print("Final Masters and PhD shapes in the normalized datasets are {} and {}.".format(normalized_masters_first_diff.shape,normalized_phd_first_diff.shape))
print("---")
### In the clustering, these normalized masters and phd datasets are again transposed.
### Determining the best k for clustering.
### With preprocessing, the silhoueete scores decrease.
### if a first diff dataset with drop=True,k should be less than equal to 9.
k=10
print("Masters data clustering results.")
print("---")

for cluster in range(2,k+1):
	
	### decorator applied with the first approach 
	### (without passing to clustering decorator manually)
	res=clustering(normalized_master_limited,nclusters=cluster,distance_metric="softdtw",pca=False) # preprocess="min_max"
	print("The silhouette score for {} clusters is {}.".format(cluster,round(res["silhouette"],3)))


print("---")
print("PhD data clustering results.")
print("---")


for cluster in range(2,k+1):	
	
	res=clustering(normalized_phd_limited,nclusters=cluster,distance_metric="softdtw",pca=False) 
	print("The silhouette score for {} clusters is {}.".format(cluster,round(res["silhouette"],3)))

time.sleep(5)

### Optimal clusters with silhouette scores with normalized datasets.
vis=True #for silhouette visualization.

print("---")

### decorator call with arguments, for both master's and phd,second approach.
### Metrics do not need to be the same, but the same metric in genereal produces higher results.
res=clustering_decorator(vis,visualize_silhoueete,
	distance_metric="softdtw")(clustering)(normalized_masters_first_diff,
	nclusters=2,plot=True,distance_metric="softdtw",title="Adjusted Time Series: Master's Dataset",pca=False)


rd=res["dict_of_cluster_names"]

for cluster,columns in rd.items():
	print("For Masters :", cluster,"has {} series.".format(len(columns)))

print("---")

res=clustering_decorator(vis,visualize_silhoueete,distance_metric="softdtw")(clustering)(normalized_phd_first_diff,nclusters=3,
	plot=True,distance_metric="softdtw",title="Adjusted Time Series: PhD Dataset",pca=False)

rd=res["dict_of_cluster_names"]

for cluster,columns in rd.items():
	print("For PhD :", cluster,"has {} series.".format(len(columns)))	

