###  Main procedure.
###  Check decomposition, stationarity. -✓
###  Check decompostion, stationarity relation, do series transformation if needed. -✓ 
###  Check dim reduction.
###  Check standardization.-✓
###  Check iteratively selecting ks. -✓
###  Evaluete using silhouette scores, and visualize. -✓

from models import *
from data import *
from times_series import *
import re
from sklearn.preprocessing import MinMaxScaler

masters_data=read_csv("datasets/masters.csv")
phd_data=read_csv("datasets/phd.csv")

### removes trailing space before the comma.
masters_data.columns=[re.sub('\\s*([,])s*', r'\1', mcol) for mcol in masters_data.columns]
phd_data.columns=[re.sub('\\s*([,])s*', r'\1', pcol) for pcol in phd_data.columns]

### removes trailing multiple spaces in the string.
masters_data.columns=[re.sub(' +', ' ', mcol) for mcol in masters_data.columns]
phd_data.columns=[re.sub(' +', ' ', pcol) for pcol in phd_data.columns]

master_cols=["Number of women entrants, persons","Number of men entrants, persons",
"Number of state order training women entrants, persons","Number of state order training men entrants, persons",
"Number of paid training women entrants, persons","Number of paid training men entrants, persons",
"Number of woman students, persons","Number of man students, persons","Number of state order training woman students, persons",
"Number of state order training man students, persons","Number of paid training woman students, persons",
"Number of paid training man students, persons","Number of woman graduates, persons","Number of man graduates, persons",
"Number of foreign woman entrants, persons","Number of foreign man entrants, persons","Number of foreign woman students, persons","Number of foreign man students, persons",
"Number of state order training foreign woman students, persons","Number of state order training foreign man students, persons",
"Number of paid training foreign woman students, persons","Number of paid training foreign man students, persons"]

### converts men, women words to match the ones in the phd_data.
m_cols=[]
for i in range(len(master_cols)):
	s=" "
	l=[]
	for index,spl in enumerate(master_cols[i].split()):
		
		if spl=="women":
			spl="woman"	
		elif spl=="men":
			spl="man"

		l.append(spl)
	s= s.join(i for i in l)
	m_cols.append(s)

master_limited,phd_limited=pd.DataFrame(),pd.DataFrame()

for  mcol in master_cols: 
	master_limited[mcol]=masters_data[mcol]

master_cols=m_cols

diff_list=[] # finds the columns in the masters_data which are not present in the phd data.
for i in master_cols:
	if i not in phd_data.columns:
		diff_list.append(i)


phd_cols=[i for i in master_cols if i not in diff_list]
phd_cols.append(["Men who had defended thesis, persons","Women who had defended thesis, persons"])

for pcol in phd_cols:
	phd_limited[pcol]=phd_data[pcol]


#print("Number of columns in the masters data is",len(master_limited.columns))
#print("Number of columns in the phd data is", len(phd_limited.columns))

### Most of the series are hightl non-stationary.
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
	
### The variables are non stationary.
print("Stationarity test for the Masters Dataset")
#print(decompose_and_test_stationarity(master_limited))

print("Stationarity test for the PhD Dataset")
#print(decompose_and_test_stationarity(phd_limited))


### Differencing the series to period 1 to remove stationarity.
masters_diff=master_limited.diff().dropna()
phd_diff=phd_limited.diff().dropna()


print("Number of observations in Masters before and after differencing is {} and {}".format(master_limited.shape[0],masters_diff.shape[0]))
print("Number of observations in PhD before and after differencing is are {} and {}".format(phd_limited.shape[0],phd_diff.shape[0]))


### Most of the series become stationary, but negative values are introduced.
### We check for the columns which still are non-stationary and conduct a second differencing.
### We loose observations, and need to drop a row for the stationary columns as well to be able to proceed with calculations.
### To account for negative values and for scaling, we do a Min-Max normalization.


print("Stationarity test for the Masters Dataset after Differencing")
resm=decompose_and_test_stationarity(masters_diff)
print(resm)


### Finding non-stationary columns after 1st differencing.
resm_list=[]
for i,value in enumerate(resm.values()):
	if value >0.05:
		resm_list.append(i)

print(resm_list)

print("Stationarity test for the PhD Dataset after Differencing")
resp=decompose_and_test_stationarity(phd_diff)

### Finding non-stationary columns after 2nd differencing.

print("Master's DataSet subset after 2nd differencing")

resp_list=[]
for i,value in enumerate(resp.values()):
	if value >0.05:
		resp_list.append(i)

subset=masters_diff.iloc[:,resm_list]
subset=subset.diff().dropna() # 2002 year  is dropped.



print(decompose_and_test_stationarity(subset)) # only 1 column is not stationary after 2nd differencing.

masters_diff.drop(masters_diff.columns[resm_list],axis=1,inplace=True)
masters_diff=masters_diff.merge(subset,right_index=True,left_index=True) # merging stationary columns.

print("PhD's DataSet subset after 2nd differencing")

subsetp=phd_diff.iloc[:,resp_list]
subsetp=subsetp.diff().dropna()

print(decompose_and_test_stationarity(subsetp)) # three columns are still non-stationary.

phd_diff.drop(phd_diff.columns[resp_list],axis=1,inplace=True)
phd_diff=phd_diff.merge(subsetp,right_index=True,left_index=True) # merging stationary- columns.

print("Final Master's and PhD Shapes.")
print(masters_diff.shape,phd_diff.shape)

### Column wise min-max normalization from 0 to 1.
### The decomposition results stay the same after normalization.
### Normalization does not change stationarity, so 2nd differencing was needed.

normalized_masters_diff=(masters_diff-masters_diff.min())/(masters_diff.max()-masters_diff.min())
#print(decompose_and_test_stationarity(normalized_masters_diff))

normalized_phd_diff=(phd_diff-phd_diff.min())/(phd_diff.max()-phd_diff.min())
#print(decompose_and_test_stationarity(normalized_phd_diff))

print("Masters columns and PhD columns in the normalize dataset are {} {}".format(normalized_masters_diff.shape[1],normalized_phd_diff.shape[1]))

### Determining the best k for clustering.
### With preprocessing, the silhoueete scores decrease.

k=10
print("Masters data clustering results")
for cluster in range(2,k+1):
	res=clustering(normalized_masters_diff,ncols=22,nclusters=cluster) # preprocess="min_max"
	print("The silhouette score for {} clusters is {}".format(cluster,res["silhouette"]))

print("----------")
print("PhD data clustering results")
for cluster in range(2,k+1):
	res=clustering(normalized_phd_diff,ncols=20,nclusters=cluster) # preprocess="min_max"
	print("The silhouette score for {} clusters is {}".format(cluster,res["silhouette"]))


### Optimal clusters with silhouette scores with normalized datasets.
print("Master's Data Evaluation")
res=clustering(normalized_masters_diff,ncols=22,nclusters=2,plot=True,title="Master's Experiment")
visualize_silhoueete(res["model"],res["two_dim_data"])

print("PhD Data Evaluation")
res=clustering(normalized_phd_diff,ncols=20,nclusters=2,plot=True,title="PhD Experiment")
visualize_silhoueete(res["model"],res["two_dim_data"])

