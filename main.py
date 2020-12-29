from models import *
from data import *
from times_series import *
import re


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

#Clustering
clustering(master_limited,ncols=22,nclusters=3)
clustering(phd_limited,ncols=20,nclusters=3)
