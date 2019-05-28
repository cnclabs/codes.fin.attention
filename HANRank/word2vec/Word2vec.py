import logging
import pandas as pd
from gensim.models import word2vec
import gensim

#Generate Embedding Matrix of each year
df = pd.read_pickle('../data.pkl')
df['year'] = df['year'].astype(int)
for year in range(2001,2014):
	print(year)
	data = df[df['year']>=year-5]
	data  = df[df['year']<year]	
	print(data.shape)
	datalist = data.porter_stop.tolist()
	sents = ' '.join(datalist)
	sentlist = sents.split('. ')
	print(len(sentlist))
	#sentlist = pd.read_pickle('../'+str(year)+'sent.pkl')
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = gensim.models.Word2Vec([s.split() for s in sentlist],min_count=1, size=300, window=5,workers=10)
	model.wv.save_word2vec_format(str(year)+'word2vec.txt', binary=False)

