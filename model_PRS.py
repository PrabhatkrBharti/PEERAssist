#requirements {  tqdm , pandas , numpy , tensorflow , keras , sklearn , nltk , matplotlib , mlxtend }
import sys
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
import time
from copy import deepcopy
import random
from tensorflow.keras.layers import Conv2D, Permute , MaxPooling2D, TimeDistributed, AveragePooling2D, Lambda, Softmax, Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, Input, Reshape, concatenate, dot, Multiply, RepeatVector
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

 
seed = 54321
 
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
 
print(tf.__version__)

if len(sys.argv)<2:
	print("ERROR : please provide year in command line argument")
	exit()

year = int(sys.argv[1])

if (year<2017) | (year>2020):
	print("ERROR : please provide a valid year in command line argument")
	exit()

year = str(year)


#hyper-params
paper_embed_dim = 768
rev_embed_dim = 512
 
balance_factor = 1

batch_sz = 32     #keep it even
 
lstm_base = 256

num_epochs = 20
if len(sys.argv)>2:
	num_epochs = int(sys.argv[2])
 
hyper_factor_and_dropout = 0.7
dropout = hyper_factor_and_dropout
 
activation_func = "tanh"



paperEmbed = pd.read_csv("./DATA/data/EDATA"+year+".csv")
paperEmbed = paperEmbed[["col"+str(i) for i in range(paper_embed_dim)]]
paperEmbed = paperEmbed.values.tolist()
paperEmbed = paperEmbed + [[0.0 for i in range(paper_embed_dim)]]
paper_num_tokens = len(paperEmbed)
paperMeta = pd.read_csv("./DATA/Dmeta"+year+".csv")
paperMeta = paperMeta[['paper-start' , 'paper-end']]
scMeta = pd.read_csv("./DATA/PMETA"+year+"_SCSUMMA.csv")
scMeta = scMeta[['id' , 'section-start' , 'section-end']]
print("paper_num_tokens : ",paper_num_tokens)


revEmbed = pd.read_csv("./DATA/data/REV_USE"+year+".csv")
revEmbed = revEmbed[["col"+str(i) for i in range(rev_embed_dim)]]
revEmbed = revEmbed.values.tolist()
revEmbed = revEmbed + [[0.0 for i in range(rev_embed_dim)]]
rev_num_tokens = len(revEmbed)
revMeta = pd.read_csv("./DATA/REV"+year+".csv")
revMeta = revMeta[['rev-start' , 'rev-end']]
abMeta = pd.read_csv("./DATA/AB"+year+".csv")
abMeta = abMeta[['id' , 'verdict', 'revMeta-start' , 'revMeta-end']]
num_samples = len(abMeta)
print("rev_num_tokens : ",rev_num_tokens)


num_samples = len(scMeta)
pids = []
scid = {}
for i in range(num_samples):
	pids.append(str(scMeta['id'][i]))
	scid[str(scMeta['id'][i])] = i
pids.sort()
dummy = []
for i in range(num_samples):
	j = scid[pids[i]]
	dummy.append([scMeta['id'][j] , scMeta['section-start'][j] , scMeta['section-end'][j]])
dummy = pd.DataFrame(dummy)
dummy.columns = ['id' , 'section-start' , 'section-end']
scMeta = dummy
	
print("paperMeta-revise...\n")
num_samples = len(abMeta)
paperId = {}
for i in range(num_samples):
	paperId[abMeta['id'][i]] = i

dummy = []
nf = num_samples - len(scMeta)
num_samples = len(scMeta)
for i in range(num_samples):
	idx = paperId[scMeta['id'][i]]
	dummy.append([abMeta["id"][idx], abMeta["verdict"][idx] , abMeta["revMeta-start"][idx] , abMeta["revMeta-end"][idx]])
dummy = pd.DataFrame(dummy)
dummy.columns = ['id' , 'verdict' , 'revMeta-start' , 'revMeta-end']

abMeta = dummy
num_samples = len(abMeta)
dummy = []

print("number of samples : ",num_samples)


revF = pd.read_csv("./DATA/flattered/REV"+year+".csv")
revF = revF[['text']]

analyzer = SentimentIntensityAnalyzer()
ln = len(revF)
vader = []
for i in range(ln):
	txt = revF['text'][i] + " . "
	polar = analyzer.polarity_scores(txt)
	vader.append([polar['neg'],polar['pos'],polar['neu'],polar['compound']])

paper_mx_sections = 0
paper_mx_snts = 0
rev_mx_snts = 0
num_samples = len(scMeta)
for i in range(num_samples):
	paper_mx_sections = max(paper_mx_sections , scMeta['section-end'][i]-scMeta['section-start'][i])
	for j in range(scMeta['section-start'][i] , scMeta['section-end'][i]):
		paper_mx_snts = max(paper_mx_snts , paperMeta['paper-end'][j]-paperMeta['paper-start'][j])
	for j in range(abMeta['revMeta-start'][i] , abMeta['revMeta-end'][i]):
		rev_mx_snts = max(rev_mx_snts , revMeta['rev-end'][j]-revMeta['rev-start'][j])
print("max sections , paper max sentences , review max sentences : ",paper_mx_sections ," : ",paper_mx_snts , " : ",rev_mx_snts)

xall = [i for i in range(num_samples)]
yall = [float(abMeta['verdict'][i]) for i in range(num_samples)]
xtr , xte , ytr , yte = train_test_split(xall,yall, test_size=0.25, random_state=seed, stratify=yall)
teLen = len(yte)
trLen = len(ytr)

paperTest = []
revTest = []
Ytest = []
vaderTest = []
zeroSection = [(paper_num_tokens-1) for i in range(paper_mx_snts)]

for i in xte:
	paperDummy = []
	for j in range(scMeta['section-start'][i] , scMeta['section-end'][i]):
		dummy = []
		for k in range(paperMeta['paper-start'][j] , paperMeta['paper-end'][j]):
			dummy.append(k)
		times = paper_mx_snts - (paperMeta['paper-end'][j] - paperMeta['paper-start'][j])
		for t in range(times):
			dummy.append(paper_num_tokens-1)
		paperDummy.append(dummy)
	times = paper_mx_sections - (scMeta['section-end'][i] - scMeta['section-start'][i])
	for t in range(times):
		paperDummy.append(zeroSection)
	paperTest.append(paperDummy)
	revDummy = []
	vaderDummy = []
	for j in range(abMeta['revMeta-start'][i] , abMeta['revMeta-end'][i]):
		dummy = []
		vdummy = []
		for k in range(revMeta['rev-start'][j] , revMeta['rev-end'][j]):
			dummy.append(k)
			vdummy.append(vader[k])
		times = rev_mx_snts - (revMeta['rev-end'][j] - revMeta['rev-start'][j])
		for t in range(times):
			dummy.append(rev_num_tokens-1)
			vdummy.append([0.0,0.0,1.0,0.0])
		revDummy.append(dummy)
		vaderDummy.append(vdummy)
	revTest.append(revDummy)
	vaderTest.append(vaderDummy)

Ytest = yte
print("Test-set length : ",teLen , " : ",len(paperTest) , " : ",len(revTest) , " : ", len(Ytest), " : ", len(vaderTest))


cnt = [[] , []]
for i in range(trLen):
	cnt[int(ytr[i])].append(xtr[i])
print(len(cnt[0]) , " : ",len(cnt[1]))
xtr = []
ytr = []
flag = [0 , 0]
tmp = [0 , 0]
while (flag[0]<balance_factor) | (flag[1]<balance_factor):
	times = batch_sz//2
	for i in range(2):
		for t in range(times):
			xtr.append(cnt[i][tmp[i]])
			ytr.append(float(i))
			tmp[i] += 1
			if tmp[i]==len(cnt[i]):
				tmp[i]=0
				flag[i] += 1
				random.shuffle(cnt[i])
trLen = len(ytr)
print("Train-set length : ",trLen ," : ", len(xtr))

paperTrain = []
revTrain = []
Ytrain = []
vaderTrain = []
zeroSection = [(paper_num_tokens-1) for i in range(paper_mx_snts)]

for i in xtr:
	paperDummy = []
	for j in range(scMeta['section-start'][i] , scMeta['section-end'][i]):
		dummy = []
		for k in range(paperMeta['paper-start'][j] , paperMeta['paper-end'][j]):
			dummy.append(k)
		times = paper_mx_snts - (paperMeta['paper-end'][j] - paperMeta['paper-start'][j])
		for t in range(times):
			dummy.append(paper_num_tokens-1)
		paperDummy.append(dummy)
	times = paper_mx_sections - (scMeta['section-end'][i] - scMeta['section-start'][i])
	for t in range(times):
		paperDummy.append(zeroSection)
	paperTrain.append(paperDummy)
	revDummy = []
	vaderDummy = []
	for j in range(abMeta['revMeta-start'][i] , abMeta['revMeta-end'][i]):
		dummy = []
		vdummy = []
		for k in range(revMeta['rev-start'][j] , revMeta['rev-end'][j]):
			dummy.append(k)
			vdummy.append(vader[k])
		times = rev_mx_snts - (revMeta['rev-end'][j] - revMeta['rev-start'][j])
		for t in range(times):
			dummy.append(rev_num_tokens-1)
			vdummy.append([0.0,0.0,1.0,0.0])
		revDummy.append(dummy)
		vaderDummy.append(vdummy)
	revTrain.append(revDummy)
	vaderTrain.append(vaderDummy)
Ytrain = ytr
print("Train-set length : ",trLen , " : ",len(paperTrain) , " : ",len(revTrain) , " : ", len(Ytrain) , " : ", len(vaderTrain))

Ytrain = [[0.0 , 0.0] for i in range(trLen)]
for i in range(trLen):
	if ytr[i]==1.0:
		Ytrain[i][1] = 1.0
	if ytr[i]==0.0:
		Ytrain[i][0] = 1.0

Ytest = [[0.0 , 0.0] for i in range(teLen)]
for i in range(teLen):
	if yte[i]==1.0:
		Ytest[i][1] = 1.0
	if yte[i]==0.0:
		Ytest[i][0] = 1.0

print("train-test split : ",len(Ytrain) , " : ", len(Ytest))


#numpy
trLen = len(Ytrain)
teLen = len(Ytest)
paperTrain = np.asarray(paperTrain).reshape(trLen,paper_mx_sections,paper_mx_snts)
revTrain = np.asarray(revTrain).reshape(trLen,3,rev_mx_snts)
vaderTrain = np.asarray(vaderTrain).reshape(trLen,3,rev_mx_snts,4)
Ytrain = np.asarray(Ytrain).reshape(trLen,2)

paperTest = np.asarray(paperTest).reshape(teLen,paper_mx_sections,paper_mx_snts)
revTest = np.asarray(revTest).reshape(teLen,3,rev_mx_snts)
vaderTest = np.asarray(vaderTest).reshape(teLen,3,rev_mx_snts,4)
Ytest = np.asarray(Ytest).reshape(teLen,2)



#modelling..............
def rmse_fun(pred,actual):
	return np.sqrt(np.mean((pred-actual)**2))
 
@tf.function
def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
 
lstm_out = int(lstm_base * hyper_factor_and_dropout) + int(lstm_base * 0.2)
lstm_out2 = int(lstm_base * hyper_factor_and_dropout)
num_filters = int(((3*rev_mx_snts)//7) * hyper_factor_and_dropout)
total_out = (3*7*num_filters + 3*4*rev_mx_snts) + (3*2*lstm_out) + (3*rev_embed_dim + 3*4) + (2*lstm_out2)
d_out = int((total_out//15) * hyper_factor_and_dropout)

#architecture

#paper - part
paper_input = Input(shape=(paper_mx_sections,paper_mx_snts))
paper_embedded = Embedding(paper_num_tokens, paper_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(paperEmbed), trainable=False)(paper_input)

#section - matrix
bilstm = TimeDistributed(Bidirectional(LSTM(lstm_out)))(paper_embedded)

#hierarchical bilstm
p = Bidirectional(LSTM(lstm_out2))(bilstm)
p_final = Dropout(dropout)(p)


#sentiment input
vader_input = Input(shape=(3,rev_mx_snts,4))
vader_flat = TimeDistributed(Flatten())(vader_input)

#review - part
rev_input = Input(shape=(3,rev_mx_snts))
rev_embedded = Embedding(rev_num_tokens, rev_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(revEmbed), trainable=False)(rev_input)

R = Reshape((3,rev_mx_snts,rev_embed_dim,1))(rev_embedded)

r_a = TimeDistributed(Conv2D(num_filters, (1,rev_embed_dim),activation=activation_func))(R)
r_a = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts,1)))(r_a)
r_a = TimeDistributed(Flatten())(r_a)

r_b = TimeDistributed(Conv2D(num_filters, (2,rev_embed_dim),activation=activation_func))(R)
r_b = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-1,1)))(r_b)
r_b = TimeDistributed(Flatten())(r_b)

r_c = TimeDistributed(Conv2D(num_filters, (3,rev_embed_dim),activation=activation_func))(R)
r_c = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-2,1)))(r_c)
r_c = TimeDistributed(Flatten())(r_c)

r_d = TimeDistributed(Conv2D(num_filters, (4,rev_embed_dim),activation=activation_func))(R)
r_d = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-3,1)))(r_d)
r_d = TimeDistributed(Flatten())(r_d)

r_e = TimeDistributed(Conv2D(num_filters, (5,rev_embed_dim),activation=activation_func))(R)
r_e = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-4,1)))(r_e)
r_e = TimeDistributed(Flatten())(r_e)

r_f = TimeDistributed(Conv2D(num_filters, (6,rev_embed_dim),activation=activation_func))(R)
r_f = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-5,1)))(r_f)
r_f = TimeDistributed(Flatten())(r_f)

r_g = TimeDistributed(Conv2D(num_filters, (7,rev_embed_dim),activation=activation_func))(R)
r_g = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-6,1)))(r_g)
r_g = TimeDistributed(Flatten())(r_g)

r = concatenate([r_a , r_b , r_c , r_d , r_e , r_f , r_g , vader_flat] , axis=-1)
r_final = Flatten()(r)
r_final = Dropout(dropout)(r_final)

#rev-attended-paper
ra = TimeDistributed(RepeatVector(paper_mx_sections))(r)

rp_p = TimeDistributed(RepeatVector(3))(bilstm)
rp_p = Permute((2,1,3))(rp_p)

rpa = concatenate([ra , rp_p] , axis=-1)
rpaV = TimeDistributed(TimeDistributed(Dense(1,activation="tanh")))(rpa)
rpaV = TimeDistributed(Flatten())(rpaV)
rpaV_out = TimeDistributed(Softmax())(rpaV)
rpav = TimeDistributed(RepeatVector(2*lstm_out))(rpaV_out)
rpav = Permute((1,3,2))(rpav)

rp = Multiply()([rpav , rp_p])
rp = Reshape((3,paper_mx_sections,2*lstm_out,1))(rp)

rp = TimeDistributed(AveragePooling2D(pool_size=(paper_mx_sections,1)))(rp)
rp = TimeDistributed(Dropout(dropout))(rp)
rp = Flatten()(rp)


#paper-attended-reviews
pa = RepeatVector(3)(p)
pa = TimeDistributed(RepeatVector(rev_mx_snts))(pa)

Rsent = concatenate([rev_embedded , vader_input] , axis=-1)

pra = concatenate([pa , Rsent] , axis=-1)
praV = TimeDistributed(TimeDistributed(Dense(1,activation="sigmoid")))(pra)
praV_out = TimeDistributed(Flatten())(praV)
prav = TimeDistributed(RepeatVector(rev_embed_dim+4))(praV_out)
prav = Permute((1,3,2))(prav)

pr = Multiply()([prav , Rsent])
pr = Reshape((3,rev_mx_snts,rev_embed_dim+4,1))(pr)

pr = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts,1)))(pr)
pr = TimeDistributed(Dropout(dropout))(pr)
pr = Flatten()(pr)


#MLP
mlp_inp = concatenate([r_final, rp , pr , p_final] , axis=-1)


dim1 = Dense(d_out,activation="relu")(mlp_inp)
dim1 = Dropout(dropout)(dim1)
dim2 = Dense(d_out,activation="softsign")(mlp_inp)
dim2 = Dropout(dropout)(dim2)
dim3 = Dense(d_out,activation="sigmoid")(mlp_inp)
dim3 = Dropout(dropout)(dim3)

dim = concatenate([dim1 , dim2 , dim3] , axis=-1)

mlp = Dense(2,activation="linear")(dim)
mlp_out = Softmax()(mlp)


model = Model(inputs=[paper_input, rev_input , vader_input] , outputs=mlp_out)

model.compile(optimizer = "adam", loss = "binary_crossentropy" ,metrics =["accuracy"])

los = 999.0
acc = 0.0
compare_val = -999
y_predict = []

for i in range(num_epochs):
	print("Epoch [",i+1,"/",num_epochs,"] :")
	model.fit([paperTrain , revTrain , vaderTrain], Ytrain, batch_size = batch_sz, shuffle=False,epochs = 1,verbose=2)
	tmp = model.predict([paperTest , revTest , vaderTest])
	#
	mat = [[0,0],[0,0]]
	for i in range(teLen):
		yp = 0
		yt = 0
		if tmp[i][1]>tmp[i][0]:
			yp=1
		if Ytest[i][1]>Ytest[i][0]:
			yt=1
		mat[yt][yp] += 1
	#
	prec = [mat[0][0]/max(1,mat[0][0]+mat[1][0]) , mat[1][1]/max(1,mat[1][1],mat[0][1])]
	rec = [mat[0][0]/max(1,mat[0][0]+mat[0][1]) , mat[1][1]/max(1,mat[1][1]+mat[1][0])]
	f1 = [(2*prec[0]*rec[0])/max(0.001,prec[0]+rec[0]) , (2*prec[1]*rec[1])/max(0.001,prec[1]+rec[1])]
	#
	val = (mat[0][0]+mat[1][1])/teLen
	#
	if val>compare_val:
		y_predict = deepcopy(tmp)
		los,acc = model.evaluate([paperTest , revTest , vaderTest] , Ytest, batch_size=batch_sz , verbose=0)
		compare_val = val

print("test-set [ loss : ",los,"  , accuracy : ",acc," ]")

print("confusion matrix..............")
yt = []
yp = []
mat = [[0,0],[0,0]]
for i in range(teLen):
	if y_predict[i][0]>=y_predict[i][1]:
		yp.append(0)
	else:
		yp.append(1)
	if Ytest[i][0]>=Ytest[i][1]:
		yt.append(0)
	else:
		yt.append(1)
	mat[yt[i]][yp[i]] += 1
print("confusion_matrix : ",mat)
yp = np.asarray(yp).reshape(teLen)
yt = np.asarray(yt).reshape(teLen)
print(metrics.classification_report(yt,yp))

def plot_conf_mat(cm , xlen, ylen , title="confusion matrix", cmap = plt.cm.Blues):
	fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(2, 2) , cmap=cmap , colorbar=True , show_normed=True, show_absolute=False)
	plt.xlabel('Predicted Label', fontsize=10, labelpad=15)
	plt.ylabel('True Label', fontsize=10,labelpad=15)
	marks = np.arange(2)
	label = ["REJ" , "ACC"]
	plt.xticks(marks, label)
	plt.yticks(marks, label)
	plt.title(title, fontsize=14 , pad=2)
	plt.show()

conf_matrix = np.asarray(mat).reshape(2,2)
plot_conf_mat(conf_matrix , 2, 2 , title="")

