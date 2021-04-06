#requirements: {pandas, json , os}
import pandas as pd
import os
from preprocess import sentence_tokenizer as stok
import json

print("No. of papers (year-wise) ========")
yearId = 2017
num_papers = 0
while yearId<=2020:
	year = str(yearId)
	papers = [f for f in os.listdir("./raw/"+year+"/") if f[-11:]==".paper.json"]
	num_papers += len(papers)
	print(year," : ",len(papers))
	yearId += 1

print("\nNo. of reviews (year-wise) ========")
yearId = 2017
while yearId<=2020:
	year = str(yearId)
	revs = [f for f in os.listdir("./raw/"+year+"/") if ((f[-11:]!=".paper.json") & (f[-5:]==".json"))]
	print(year," : ",3*len(revs))
	yearId += 1

print("\nAcceptance rate (year-wise) ========")
num_accept = [0,0,0,0]
num_reviews = [0,0,0,0]
num_sentences = [0,0,0,0]
num_words = [0,0,0,0]
yearId = 2017
while yearId<=2020:
	year = str(yearId)
	revs = [f for f in os.listdir("./raw/"+year+"/") if ((f[-11:]!=".paper.json") & (f[-5:]==".json"))]
	num_acc = 0
	num_revs = 0
	num_snts = 0
	num_wds = 0
	for rev in revs:
		fl = open("./raw/"+year+"/"+rev)
		data = json.load(fl)
		if data['verdict']=="Accept":
			num_acc += 1
		for dummy in data['reviews']:
			if dummy['comments']!="":
				num_revs += 1
				snts = stok.sentence_tokenize(dummy['comments'])
				num_snts += len(snts)
				for st in snts:
					num_wds += len(st[0].split(" "))
	num_accept[yearId-2017] = num_acc
	num_reviews[yearId-2017] = num_revs
	num_sentences[yearId-2017] = num_snts
	num_words[yearId-2017] = num_wds
	print(year," : total rev_ids : ",num_revs/3," Acc-rate : ",(3*num_acc)/num_revs)
	yearId += 1

print("\nLength of reviews (in sentences) (year-wise) ========")
yearId = 2017
while yearId<=2020:
	print(yearId," : ",num_sentences[yearId-2017]/num_reviews[yearId-2017])
	yearId += 1
	
print("\nLength of reviews (in words) (year-wise) ========")
yearId = 2017
while yearId<=2020:
	print(yearId," : ",num_words[yearId-2017]/num_reviews[yearId-2017])
	yearId += 1
	
print("ALL DATASET")
print("number of papers : ",num_papers)
total_revs = num_reviews[0]+num_reviews[1]+num_reviews[2]+num_reviews[3]
print("number of reviews : ",total_revs)
print("Acceptance rate : ",(num_accept[0]+num_accept[1]+num_accept[2]+num_accept[3])/num_papers)
print("length of reviews (in sentences) : ",(num_sentences[0]+num_sentences[1]+num_sentences[2]+num_sentences[3])/total_revs)
print("length of reviews (in words) : ",(num_words[0]+num_words[1]+num_words[2]+num_words[3])/total_revs)
