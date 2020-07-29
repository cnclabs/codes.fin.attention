import pandas as pd
from nltk.corpus import stopwords
from multiprocessing import Pool
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def f(x):
	stopWords = set(stopwords.words('english'))
	porter_stemmer = PorterStemmer()
	textlist = x['doclist'].tolist()
	newlist = []
	for a in textlist:
		new_sent = ''
		doc = a.split('. ')
		for sents in doc:		
			words  = sents.split(' ')
			for word in words:
				if word not in stopWords or word=='.':
					if len(word)>1:
						word = porter_stemmer.stem(word)
						new_sent = new_sent + word + ' '
			new_sent = new_sent.strip()
			new_sent = new_sent +'. '
		newlist.append(new_sent)
		print(len(newlist))
	print(len(newlist))
	se = pd.Series(newlist)
	x['porter_stop'] = se.values
	return x


#Remove stopwords (nltk) and Porter stemming
if __name__ == '__main__':
	p = Pool(10)
	df = pd.read_pickle('./data.pkl')
	print(df)	
	tendf = np.array_split(df,10)
	print(tendf[3])
	result = p.map(f, [tendf[0], tendf[1], tendf[2], tendf[3],tendf[4], tendf[5],tendf[6], tendf[7],tendf[8], tendf[9]])
	p.close()
	final = pd.concat(list(result))
	print(final)
	final = pd.DataFrame(final)
	final.to_pickle('./data.pkl')

