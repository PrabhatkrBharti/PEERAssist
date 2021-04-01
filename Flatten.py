#requirements : { tqdm , pandas , json }
from tqdm import tqdm
import pandas as pd
import json
from preprocess import sentence_tokenizer as stok


print("\n=======Loading Datas & Tokenizing=======\n")
# get mapping
df = pd.read_csv("raw/files_map.csv")
df = df[['year', 'id']]
samples = len(df)
print("num of samples : ",samples)

# max sentences year-wise
mxSnt = [[0 ,0 ,0 ,0]]
# score-wise [1-10] samples distribution in each year [2017-20]
score_cnt = [[0 for i in range(11)] for j in range(4)]
score_cnt[0][0] = 2017
score_cnt[1][0] = 2018
score_cnt[2][0] = 2019
score_cnt[3][0] = 2020
# flattered-reviews year-wise , and corresponding metadata
X = [[] , [] , [] , []]
Xmeta = [[] , [] , [] , []]
Xcnt = [0,0,0,0]
# flattered abstract year-wise , and corresponding metadata
AB = [[] , [] , [] , []]
ABmeta = [[] , [] , [] , []]
ABcnt = [0,0,0,0]

# for internal use (to check reviews and abstract are going in line and also for later use)
tmp = [0,0,0,0]
# fill all data from raw after processing 
flag = False

for i in tqdm(range(samples),desc="processing		"):
	year = str(df['year'][i])
	yearId = int(year)-2017
	sampleId = str(df['id'][i])
	rev = open("raw/"+year+"/"+sampleId+".json")
	rev = json.load(rev)
	num_revs = 0
	for review in rev['reviews']:
		try:
			score_cnt[yearId][review['recommendation']] += 1
			x = stok.sentence_tokenize(review['comments'])
			Xmeta[yearId].append([review['recommendation'] , Xcnt[yearId] , Xcnt[yearId]+len(x)])
			Xcnt[yearId] += len(x)
			X[yearId] = X[yearId] + x
			mxSnt[0][yearId] = max(mxSnt[0][yearId] , len(x))
			num_revs += 1
		except KeyError:
			continue
			
	if num_revs!=3:
		flag = True
		print("break due to no. of reviews @ ",sampleId," , year : ",year)
		break
	x = stok.sentence_tokenize(rev['abstract'])
	verdict = -1
	if rev['verdict']=='Accept':
		verdict = 1
	elif rev['verdict']=='Reject':
		verdict = 0
	else:
		flag = True
		print("break due to meta verdict @ ",sampleId," , year : ",year)
		break
	ABmeta[yearId].append([sampleId , verdict , ABcnt[yearId] , ABcnt[yearId]+len(x) , tmp[yearId] , tmp[yearId]+num_revs])
	ABcnt[yearId] += len(x)
	tmp[yearId] += num_revs
	AB[yearId] = AB[yearId] + x
	
if flag==False:	
	#print(AB[0][0])
	# save all data
	#print("max sentences year-wise [2017-20]")
	#print(mxSnt)
	#mxSnt = pd.DataFrame(mxSnt)
	#mxSnt.columns = ['2017','2018','2019','2020']
	#mxSnt.to_csv("DATA/mxSnt.csv")

	#print("score-wise [1-10] samples distribution in each year [2017-20]")
	#for sc_wise in score_cnt:
	#	print(sc_wise)
	#score_cnt = pd.DataFrame(score_cnt)
	#score_cnt.columns = ['year' , '1' , '2' ,'3' ,'4','5','6','7','8','9','10']
	#score_cnt.to_csv("DATA/score_wise_count.csv")
	
	#print("ABmeta length (year-wise) ")
	yearId = 2017
	for dummy in ABmeta:
		#print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['id' , 'verdict' , 'abstract-start' ,'abstract-end' ,'revMeta-start','revMeta-end']
		dummy.to_csv("DATA/AB"+str(yearId)+".csv")
		yearId += 1
	
	#print("Xmeta length (year-wise) ")
	yearId = 2017
	for dummy in Xmeta:
		#print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['recommendation' , 'rev-start','rev-end']
		dummy.to_csv("DATA/REV"+str(yearId)+".csv")
		yearId += 1
	
	#print("AB text length (year-wise) ")
	yearId = 2017
	for dummy in AB:
		#print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['text']
		dummy.to_csv("DATA/flattered/AB"+str(yearId)+".csv")
		yearId += 1
		
	#print("REV text length (year-wise) ")
	yearId = 2017
	for dummy in X:
		#print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['text']
		dummy.to_csv("DATA/flattered/REV"+str(yearId)+".csv")
		yearId += 1

	# for verification
	#print("ABcnt : ",ABcnt)
	#print("Xcnt : ",Xcnt)
	#print("tmp : ",tmp)



	 
