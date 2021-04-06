#requirements : { summa , tqdm , pandas }

import sys
import pandas as pd
from summa import summarizer
from tqdm import tqdm
from copy import deepcopy


if len(sys.argv)<2:
	print("ERROR : please provide year in command line argument")
	exit()

preYear = int(sys.argv[1])

if (preYear<2017) | (preYear>2020):
	print("ERROR : please provide a valid year in command line argument")
	exit()

preYear = str(preYear)



def summa_summarize(txt):
	return summarizer.summarize(txt)
  
def getSC(idx):
	ret = ""
	for k in range(paperMeta['paper-start'][idx] , paperMeta['paper-end'][idx]):
		ret += (paper['text'][k] + ". ")
	return ret

def dot_tokenize(txt):
	res = str(txt).split(".")
	result = []
	for r in res:
		if (len(r)>0) & (r!=" "):
			result.append(r)
	return result,len(result)
#print(dot_tokenize("hello there. are you there . causing any problem. "))

print("\n=======Loading Datas=======\n")
paper = pd.read_csv("./DATA/flattered/PAPER"+preYear+".csv")
paper = paper[['text']]
print("flattered paper length : ",len(paper))

paperMeta = pd.read_csv("./DATA/P"+preYear+".csv")
paperMeta = paperMeta[['paper-start','paper-end']]
print("paperMeta length : ",len(paperMeta))
scMeta = pd.read_csv("./DATA/SC"+preYear+".csv")
scMeta = scMeta[['id', 'section-start','section-end']]
print("sectionMeta length : ",len(scMeta))

print("% of paper having # of sections in the range =====\n")
mx_sec = 0
num_samples = len(scMeta)
num_sc_wise = [0 for i in range(5)]
for i in range(num_samples):
	mx_sec = max(mx_sec , scMeta['section-end'][i]-scMeta['section-start'][i])
	sc_base = min((scMeta['section-end'][i]-scMeta['section-start'][i])//10 , 4)
	num_sc_wise[sc_base] += 1

print("[0-9]	=	%.2f" % (100*(num_sc_wise[0]/num_samples))," %")
print("[10-19]	=	%.2f" % (100*(num_sc_wise[1]/num_samples))," %")
print("[20-29]	=	%.2f" % (100*(num_sc_wise[2]/num_samples))," %")
print("[30-39]	=	%.2f" % (100*(num_sc_wise[3]/num_samples))," %")
print("[REST]	=	%.2f" % (100*(num_sc_wise[4]/num_samples))," %")
print("==================================================\n")



print("\n=======Generating Sectional Summary=======\n")
mx_sec = 30 #fixed
rmx_sec = 0
X = []
sc = []
v = 0
nf = 0
n_p = 0
prv_per = -1
print("section operator : ",mx_sec)
for i in tqdm(range(num_samples) , desc="processing  "):
	num_sec = scMeta['section-end'][i] - scMeta['section-start'][i]
	if (num_sec//10)==4:
		n_p += 1
	dv = max(1 , num_sec//mx_sec)
	txt = ""
	j = scMeta['section-start'][i]
	ln = 0
	tmp = 0
	while j<scMeta['section-end'][i]:
		txt += getSC(j)
		j += 1
		tmp += 1
		if tmp==dv:
			txt = summa_summarize(txt)
			if (len(txt)>0) & (txt!=" "):
				X.append(deepcopy(txt))
				ln += 1
			else:
				nf += 1
			txt = ""
			tmp = 0
	if tmp>0:
		txt = summa_summarize(txt)
		if (len(txt)>0) & (txt!=" "):
			X.append(deepcopy(txt))
			ln += 1
		else:
			nf += 1
	sc.append([scMeta['id'][i] , v, v+ln])
	v += ln
	rmx_sec = max(rmx_sec , ln)

print("real max sections in a paper: ",rmx_sec)


sc = pd.DataFrame(sc)
sc.columns = ['id' , 'section-start' , 'section-end']

X = pd.DataFrame(X)
X.columns = ['text']

print("\n=======Saving section level Datas=======\n")
sc.to_csv("./DATA/PMETA"+preYear+"_SCSUMMA.csv")


print("% of sections having # of sentences in the range ====\n")
num_sections = len(X)
num_snt_wise = [0 for i in range(5)]
for i in range(num_sections):
	res,ln = dot_tokenize(X['text'][i])
	snt_base = min(ln//10 , 4)
	num_snt_wise[snt_base] += 1
	
print("[0-9]	=	%.2f" % (100*(num_snt_wise[0]/num_sections))," %")
print("[10-19]	=	%.2f" % (100*(num_snt_wise[1]/num_sections))," %")
print("[20-29]	=	%.2f" % (100*(num_snt_wise[2]/num_sections))," %")
print("[30-39]	=	%.2f" % (100*(num_snt_wise[3]/num_sections))," %")
print("[REST]	=	%.2f" % (100*(num_snt_wise[4]/num_sections))," %")
print("=====================================================\n")




print("\n=======Generating Sentence level Datas=======\n")
num_sections = len(X)
D = []
Dmeta = []
fxd_snt_op = 10 #fixed
mx_snts = 0
v = 0
print("sentence operator : ",fxd_snt_op)
for i in tqdm(range(num_sections),desc="processing		"):
	res,ln = dot_tokenize(X['text'][i])
	dv = max(1 , ln//fxd_snt_op)
	tmp = 0
	txt = ""
	ln = 0
	for snt in res:
		txt += (snt + " . ")
		tmp += 1
		if (tmp==dv):
			D.append(txt)
			ln += 1
			txt = ""
			tmp = 0
	if tmp>0:
		D.append(txt)
		ln += 1
	Dmeta.append([v , v+ln])
	v += ln
	mx_snts = max(mx_snts , ln)
print("real max sentences in a section : ",mx_snts)

print("\n=======Saving sentence level Datas=======\n")

Dmeta = pd.DataFrame(Dmeta)
Dmeta.columns = ['paper-start' , 'paper-end']
Dmeta.to_csv("./DATA/Dmeta"+preYear+".csv")


D = pd.DataFrame(D)
D.columns = ['text']
D.to_csv("./DATA/flattered/D"+preYear+".csv")



