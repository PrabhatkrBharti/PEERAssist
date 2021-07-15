#requirements : { tqdm , pandas , numpy , bert-for-tf2  ,  sentencepiece , tensorflow , tensorflow.keras ,  move scibert_model to directory ./scibert_model }

import sys
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig , map_stock_config_to_params , load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, AveragePooling2D, Lambda, Softmax, Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, Input, Reshape, concatenate, dot, Multiply, RepeatVector
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
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


print("\n=======Loading Datas=======\n")
D = pd.read_csv("./DATA/flattered/D"+embYear+".csv")
D = D[['text']]

num_tokens = len(D)
print(num_tokens)


#build XDATA
XDATA = []
num_samples = len(D)
for i in range(num_samples):
	XDATA.append([D['text'][i]+" . ", i%2])
	

XDATA = pd.DataFrame(XDATA)
XDATA.columns = ['text' , 'decision']

rows = len(XDATA)
xall = [i for i in range(rows)]
yall = [XDATA['decision'][i] for i in range(rows)]

xtr = xall
ytr = yall
xte = [i for i in range(5)]
yte = [XDATA['decision'][i] for i in range(5)]

XALL = []
test = []
verdict = ['REJECT' , 'ACCEPT']
rows = len(XDATA)
for i in range(rows):
	XALL.append([XDATA['text'][i] , verdict[i%2]])
XALL = pd.DataFrame(XALL)
XALL.columns = ['text' , 'decision']

rows = len(yte)
for i in range(rows):
	test.append([XDATA['text'][xte[i]] , verdict[yte[i]]])
test = pd.DataFrame(test)
test.columns = ['text' , 'decision']



#bert_model..............................................
print("\n=======Scibert Model=======\n")
bert_model_name = "scibert_scivocab_uncased"
bert_ckpt_dir = os.path.join("scibert_model/",bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir,"bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir,"bert_config.json")

print("bert chekpoint dir : ",bert_ckpt_dir)
print("bert chekpoint file : ",bert_ckpt_file)
print("bert config dir : ",bert_config_file)

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

class PaperData:
	DATA_COLUMN = "text"
	LABEL_COLUMN = "decision"
	def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=16000):
		self.tokenizer = tokenizer
		self.max_seq_len = 0
		((self.train_x, self.train_y), (self.test_x, self.test_y)) =\
			map(self._prepare, [train, test])
		self.max_seq_len = min(self.max_seq_len, max_seq_len)
		
	def _prepare(self, df):
		x, y = [], []
		for _, row in tqdm(df.iterrows()):
			text, label =\
				row[PaperData.DATA_COLUMN], row[PaperData.LABEL_COLUMN]
			tokens = self.tokenizer.tokenize(text)
			tokens = ["[CLS]"] + tokens + ["[SEP]"]
			token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
			self.max_seq_len = max(self.max_seq_len, len(token_ids))
			x.append(token_ids)
			y.append(classes.index(label))
		return np.array(x), np.array(y)
		
def pad_sbert(ids,msl):
	x = []
	for input_ids in ids:
		input_ids = input_ids[:min(len(input_ids), msl)]
		input_ids = input_ids + [0] * (msl - len(input_ids))
		x.append(np.array(input_ids))
	return np.array(x)
	
def create_model(max_seq_len, bert_ckpt_file):
	with tf.io.gfile.GFile(bert_config_file, "r") as reader:
		bc = StockBertConfig.from_json_string(reader.read())
		bert_params = map_stock_config_to_params(bc)
		bert_params.adapter_size = None
		bert = BertModelLayer.from_params(bert_params, name="bert")
	input_ids = Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
	bert_output = bert(input_ids)
	
	cls_out = Lambda(lambda seq: seq[:, 0, :])(bert_output)
	#logits = Dense(units=100, activation="tanh")(cls_out)
	#logits = Dropout(0.5)(logits)
	#logits = Dense(units=len(classes), activation="softmax")(logits)
	
	model = Model(inputs=input_ids, outputs=cls_out)
	model.build(input_shape=(None, max_seq_len))
	load_stock_weights(bert, bert_ckpt_file)
	return model
	
print("\n=======Tokenizing Sentences=======\n")
classes = XALL.decision.unique().tolist()
data = PaperData(XALL, test, tokenizer, classes, max_seq_len=1500)

num_dvs = 500
data_dvs = [[] for i in range(num_dvs)]
dataMeta = [[] for i in range(num_dvs)]
num_tokens = len(XDATA)
for i in range(num_tokens):
	data_dvs[(24+len(data.train_x[i]))//25].append(data.train_x[i])
	dataMeta[(24+len(data.train_x[i]))//25].append(i)


print("\n=======Generating Embeddings=======\n")
def tk_avg(lst):
	ln = len(lst)
	dm = len(lst[0])
	ret = [0.0 for i in range(dm)]
	for x in lst:
		for i in range(dm):
			ret[i] += x[i]
	for i in range(dm):
		ret[i] = ret[i]/ln
	return ret
	
data2embed = []
Edata = [[] for i in range(num_tokens)]
for i in tqdm(range(1,num_dvs) , desc="processing		"):
	print("Batch : [",i,"/",num_dvs,"] -------------------")
	if len(data_dvs[i])==0:
		continue
	token_sz = i*25
	flag = False
	d2eMeta = []
	data2embed = []
	st = 0
	if (i*25)>512:
		flag = True
		nt = (i*25)//512
		nt = (i*25)//(nt+1)
		for z in data_dvs[i]:
			lst = []
			c = 0
			tlen = len(z)
			for k in range(tlen):
				lst.append(z[k])
				if len(lst)==nt:
					data2embed.append(deepcopy(lst))
					c += 1
					lst = []
			if len(lst)>0:
				data2embed.append(deepcopy(lst))
				c += 1
			d2eMeta.append([st , st+c])
			st += c
		data2embed = pad_sbert(data2embed,nt)
		token_sz = nt
	else:
		data2embed = pad_sbert(data_dvs[i],i*25)
	model = create_model(token_sz, bert_ckpt_file)
	model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
	
	predict_y = model.predict(data2embed,verbose=1,batch_size=10)
	predict_y = predict_y.tolist()
	if flag==True:
		fpy = []
		for x in d2eMeta:
			lst = []
			for k in range(x[0],x[1]):
				lst.append(predict_y[k])
			lst = tk_avg(lst)
			fpy.append(deepcopy(lst))
		predict_y = deepcopy(fpy)
	ln = len(predict_y)
	for j in range(ln):
		Edata[dataMeta[i][j]] = deepcopy(predict_y[j])
	
print("\n=======Dimension Cheking=======\n")
num_tokens = len(XDATA)
print("Embedding matrix-size (num of sentences * paper_embed_dim) : [",num_tokens,"] : ",len(Edata)," * 768")
for i in range(num_tokens):
	if len(Edata[i])!=768:
		print("ERROR : Dimension uneven.")
		break

print("\n=======Saving Embedding Matrix=======\n")
Edata = pd.DataFrame(Edata)
Edata.columns = ["col"+str(i) for i in range(768)]
Edata.to_csv("./DATA/data/EDATA"+embYear+".csv")

