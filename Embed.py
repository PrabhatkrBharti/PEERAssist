#requirements : { pandas , numpy , tqdm , tensorflow , tensorflow_hub }
import sys
from tqdm import tqdm
import tensorflow as tf
from tensorflow.errors import ResourceExhaustedError
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import time
from copy import deepcopy
import random

seed = 17181920
 
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


if len(sys.argv)<2:
	print("ERROR : please provide year in command line argument")
	exit()

embYear = int(sys.argv[1])

if (embYear<2017) | (embYear>2020):
	print("ERROR : please provide a valid year in command line argument")
	exit()

embYear = str(embYear)

print("\n=======Loading USE model=======\n")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)

# USE : USE model (preloaded)
# x : list of sentences (current batch)
# index : identifier to this particular batch
# log : opened log file
def generateEmbed(x, index, log):
	flag = True
	result = []
	try:
		vect = model(x)
		result = np.array(vect).tolist()
	except ResourceExhaustedError:
		log.write(str(index)+"\n")
		print("OOM Error while embedding (USE) : batch ",index)
		flag = False
	return flag, result

batch_sz = 50

def get_batch(data, xlen, idx):
	batch = []
	iterator = idx*batch_sz
	for z in range(batch_sz):
		if iterator>=xlen:
			break
		batch.append(deepcopy(data[iterator]))
		iterator += 1
	return batch

# review embeddings
print("\n=======Generating Embeddings=======\n")
yearId = int(embYear)
flag = True
while yearId<=int(embYear):
	df = pd.read_csv("DATA/flattered/REV"+str(yearId)+".csv")
	df = df[['text']]
	X = []
	xlen = len(df)
	for i in range(xlen):
		X.append(df['text'][i])
	
	print("total sentences REV",yearId," : ",xlen," : ",len(X))

	num_batches = (xlen+batch_sz-1)//batch_sz
	
	Xembed = []
	
	with open("USElog.txt","w+") as log:
		start_time = time.time()
		for t in tqdm(range(num_batches),desc="processing		"):
			flag , vec = generateEmbed(get_batch(X , xlen , t), t, log)
			if flag==False:
				print("break @ ",t)
				break
			Xembed = Xembed + vec
	
	if flag==False:
		print(yearId,"-------- ERROR while generating embedding")
		break
	xdf = pd.DataFrame(Xembed)
	xdf.columns = ["col"+str(i) for i in range(512)]
	xdf.to_csv("DATA/data/REV_USE"+str(yearId)+".csv")
	yearId += 1
	
