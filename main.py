####  Main procedure.
# Check decomposition, stationarity. -✓
# Check decompostion, stationarity relation, do series transformation if needed.
# Check dim reduction.
# Check standardization.-✓
# Check iteratively selecting ks. -✓
# Evaluete using silhouette scores, and visualize. -✓


from models import *
from data import *
from times_series import *
import re
from sklearn.preprocessing import MinMaxScaler

masters_data=read_csv("datasets/masters.csv")
phd_data=read_csv("datasets/phd.csv")

#removes trailing space before the comma.
masters_data.columns=[re.sub('\\s*([,])s*', r'\1', mcol) for mcol in masters_data.columns]
phd_data.columns=[re.sub('\\s*([,])s*', r'\1', pcol) for pcol in phd_data.columns]

#removes trailing multiple spaces in the string.
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


# converts men, women words to match the ones in the phd_data.
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


print("Number of columns in the masters data is",len(master_limited.columns))
print("Number of columns in the phd data is", len(phd_limited.columns))


plot=True

# Most of the series are hightl non-stationary.
def decompose_and_test_stationarity(df,ncols=10):

	for index, column in enumerate(df):

		pts=TimesSeries(df[column])	
		if plot: pts.decompose().plot_decomposition()
		
		print(column)
		print(pts.test_stationarity())
		print("-------")

		if index==ncols: break



# The variables are non stationary.
print("Stationarity test for the Masters Dataset")
#decompose_and_test_stationarity(master_limited,ncols=3)

print("Stationarity test for the PhD Dataset")
#decompose_and_test_stationarity(phd_limited)


#Differencing the series to period 1 to remove stationarity.
masters_diff=master_limited.diff().dropna()
phd_diff=phd_limited.diff().dropna()

print("Number of observations in Masters after differencing is are {} and {}".format(master_limited.shape[0],masters_diff.shape[0]))
print("Number of observations in PhD after differencing is are {} and {}".format(phd_limited.shape[0],phd_diff.shape[0]))



### Most of the series become stationary, but negative values are introduced.
### To account for negative values and for scaling, we do a Min-Max normalization.


# decompose_and_test_stationarity(masters_diff,ncols=3)
# decompose_and_test_stationarity(phd_diff,ncols=3)


#  Column wise min-max normalization from 0 to 1.
normalized_masters_diff=(masters_diff-masters_diff.min())/(masters_diff.max()-masters_diff.min())
#decompose_and_test_stationarity(normalized_masters_diff,ncols=3)

normalized_phd_diff=(phd_diff-phd_diff.min())/(phd_diff.max()-phd_diff.min())
#decompose_and_test_stationarity(normalized_phd_diff,ncols=3)


print("Masters columns and PhD columns in the normalize dataset are {} {}".format(normalized_masters_diff.shape[1],normalized_phd_diff.shape[1]))

#Determining the best k for clustering.
# With preprocessing, the silhoueete scores decrease.
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


#Optimal clusters with silhouette scores with normalized datasets.
print("Master's Data Evaluation")
res=clustering(normalized_masters_diff,ncols=22,nclusters=2,plot=True)
visualize_silhoueete(res["model"],res["two_dim_data"])

print("PhD Data Evaluation")
res=clustering(normalized_phd_diff,ncols=20,nclusters=2,plot=True)
visualize_silhoueete(res["model"],res["two_dim_data"])

