import pandas as pd
import os
import numpy as np
import math
#Merge all 10-K reports to dataframe
doclist = []
docname = []
for year in range(1996,2014):
	filepath = '../dataset/mdna_pre/'+str(year)+'/'
	
	dirs = os.listdir(filepath)
	dirs.sort()
	for file in dirs:
		df = pd.read_table(filepath+file)
		y = df.values
		a = ''
		for v in y:
			for r in v:
				string = ''.join(v)
			string  = string + '. '
			a = a + string
		a = a.strip()
		doclist.append(a)
		docname.append(file)

df_doc = pd.DataFrame({'doclist':doclist,'docname':docname})
df_doc = df_doc.sort_values(by=['docname'])

#Merge the corresponding label (Post-event volatilities)

label_list = []
filepath = '../dataset/logfama/'
dirs = os.listdir(filepath)
dirs.sort()

for file in dirs:
	if file != ".ipynb_checkpoints":
		label_list.append(pd.read_table(filepath+file, sep=' ',header=-1))

df_label = pd.concat(label_list).reset_index(drop=True)
df_label.columns=['label','doc']
df_label = df_label.sort_values(by=['doc'])

final = df_doc.join(df_label)
final = final.reset_index()
del final['index']
del final['doc']
#The exponential of label (Post-event volatilities)
final['exp'] = final['label'].apply(lambda x: math.exp(x))      

#The year of the report

final['year'] = final['docname'].str[-22:-18]
final['year']=final['year'].astype(int)

#Classifiy the corresponding label (Post-event volatilities) 

frame = pd.DataFrame()
for i in range(1996,2014):
	df = final[final['year']==i]
	l = np.array(df['exp'].tolist())
	t = np.percentile(l, 30)
	s = np.percentile(l, 60)
	e = np.percentile(l, 80)
	n = np.percentile(l, 90)
	bins= [-float("inf"),t,s,e, n,float("inf")]
	print(bins)
	df['labelmark'] = pd.cut(df['exp'],bins=bins,labels=[1,2,3,4,5])
	frames = [frame, df]
	frame = pd.concat(frames)
final = frame
final = final.reset_index()
del final['index']
final.to_pickle('./data.pkl')

