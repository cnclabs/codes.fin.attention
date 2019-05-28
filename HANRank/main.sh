# Merge dataset (10-K reports and the post-event volatities) and generate data.pkl
python3 MergeCorpus.py

# Pre-processing the Corpus (add porter_stop column in data.pkl)
python3 Preprocessing.py

# Generate Word2vec Matrix of each year
python3 ./word2vec/Word2vec.py

# Make data, label and dictionary for training model (all files would be in ./data folder)
# Note that it has to add the test year (2001~2013) behind in this script (e.g. python3 Makedata.py 2001)

python3 Makedata.py 2001

# Training 
# Note that it has to add the test year (2001~2013) behind in this script (e.g. python3 Model.py 2001)
python3 Model.py 2001

