#requirements : { pandas }
import os
import pandas as pd


year = 2017
files_map = []
while year<=2020:
	lst = [f[0:len(f)-11] for f in os.listdir("./raw/"+str(year)) if f[-11:]==".paper.json"]
	lst.sort()
	for z in lst:
		files_map.append([year,z])
	year += 1
files_map = pd.DataFrame(files_map)
files_map.columns = ['year' , 'id']
files_map.to_csv("./raw/files_map.csv")



