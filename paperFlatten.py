#requirements : { tqdm , pandas , json , re }
import pandas as pd
from tqdm import tqdm
import json
from preprocess import sentence_tokenizer as stok
import re

def scanAlpha(txt):
	if txt==None:
		return "none"
	if len(txt)==0:
		return "none"
	txt = txt.lower()
	return re.sub('[^A-Za-z ]+', '', txt)

print("\n=======Loading Datas & Tokenizing=======\n")

# get mapping
df = pd.read_csv("raw/files_map.csv")
df = df[['year', 'id']]
samples = len(df)
print("num of samples : ",samples)

# different data on number of sentences year-wise
mxSnt = [[0 ,0 ,0 ,0]]
avgSnt = [[0,0,0,0]]
scSnt = [[0,0,0,0]]
avgScSnt = [[0,0,0,0]]

# flattered-papers-sentences year-wise , and corresponding section-metadata
X = [[] , [] , [] , []]
Xmeta = [[] , [] , [] , []]
Xcnt = [0,0,0,0]
# paper-metadata containing section data year-wise
Pmeta = [[] , [] , [] , []]
Pcnt = [0,0,0,0]

# for internal use 
tmp = [0,0,0,0]
# fill all data from raw after processing 
flag = False

no_sect = []
sect = 0
for i in tqdm(range(samples),desc="processing		"):
	
	year = str(df['year'][i])
	yearId = int(year)-2017
	sampleId = str(df['id'][i])
	paper = open("raw/"+year+"/"+sampleId+".paper.json")
	paper = json.load(paper)
	num_sec = 0
	try:
		dummy = paper['metadata']['sections']
		sect += 1
	except KeyError:
		no_sect += 1
		continue
	
	dummy = paper['metadata']['sections']
	if dummy==None:
		no_sect.append([year , sampleId])
		continue
		
	for section in paper['metadata']['sections']:
		try:
			head = scanAlpha(section['heading'])
			if len(head)==0:
				flag=True
				break
			x = stok.sentence_tokenize(section['text'])
			Xmeta[yearId].append([head , Xcnt[yearId] , Xcnt[yearId]+len(x)+1])
			Xcnt[yearId] += (len(x)+1)
			X[yearId] = X[yearId] + [[head]] + x
			mxSnt[0][yearId] = max(mxSnt[0][yearId] , len(x)+1)
			num_sec += 1
		except KeyError:
			continue
	if flag==True:
		print("break due to section heading @ ",sampleId," , year : ",year)
		break
	if num_sec==0:
		flag = True
		print("break due to zero sections @ ",sampleId," , year : ",year)
		break
		
	Pmeta[yearId].append([sampleId , Pcnt[yearId] , Pcnt[yearId]+num_sec])
	Pcnt[yearId] += num_sec
	scSnt[0][yearId] = max(scSnt[0][yearId] , num_sec)

#print(samples ," : ", sect, " : ", no_sect)
	

print("\n=======Saving Datas=======\n")	
if flag==False:	
	#print(X[0][0])
	for j in range(4):
		avgSnt[0][j] = len(X[j])/len(Xmeta[j])
		avgScSnt[0][j] = len(Xmeta[j])/len(Pmeta[j])
		
	#print("papers, have no section in json....")
	#print(no_sect)
	#no_sect = pd.DataFrame(no_sect)
	#no_sect.columns = ['year','sampleId']
	#no_sect.to_csv("DATA/paper_no_sect.csv")
	# save all data
	#print("max sentences in a section year-wise [2017-20]")
	#print(mxSnt)
	#mxSnt = pd.DataFrame(mxSnt)
	#mxSnt.columns = ['2017','2018','2019','2020']
	#mxSnt.to_csv("DATA/paper_mxSnt.csv")
	
	#print("avg sentences in a section year-wise [2017-20]")
	#print(avgSnt)
	#avgSnt = pd.DataFrame(avgSnt)
	#avgSnt.columns = ['2017','2018','2019','2020']
	#avgSnt.to_csv("DATA/paper_avgSnt.csv")
	
	#print("number of sections year-wise [2017-20]")
	#print(scSnt)
	#scSnt = pd.DataFrame(scSnt)
	#scSnt.columns = ['2017','2018','2019','2020']
	#scSnt.to_csv("DATA/paper_scSnt.csv")
	
	#print("avg number of sections year-wise [2017-20]")
	#print(avgScSnt)
	#avgScSnt = pd.DataFrame(avgScSnt)
	#avgScSnt.columns = ['2017','2018','2019','2020']
	#avgScSnt.to_csv("DATA/paper_avgScSnt.csv")
	
	print("Pmeta length (year-wise) ")
	yearId = 2017
	for dummy in Pmeta:
		print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['id' , 'section-start' ,'section-end']
		dummy.to_csv("DATA/SC"+str(yearId)+".csv")
		yearId += 1
	
	print("Xmeta length (year-wise) ")
	yearId = 2017
	for dummy in Xmeta:
		print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['heading' , 'paper-start','paper-end']
		dummy.to_csv("DATA/P"+str(yearId)+".csv")
		yearId += 1
		
	print("Paper text length (year-wise) ")
	yearId = 2017
	for dummy in X:
		print(yearId," -th length : ",len(dummy))
		dummy = pd.DataFrame(dummy)
		dummy.columns = ['text']
		dummy.to_csv("DATA/flattered/PAPER"+str(yearId)+".csv")
		yearId += 1

	# for verification
	print("ABcnt : ",Pcnt)
	print("Xcnt : ",Xcnt)



	 
