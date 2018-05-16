'''
Converts text strings to integer sequences,
mostly for use with NLMs and other models with RNNs
'''
import pandas as pd
import numpy as np
import h5py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from tools.generic import one_hot
from tools.text import doc_to_char, pad_integers, to_integer

# Importing the text files
records = pd.read_csv('~/data/syndromic/good_cc_records.csv',
                      usecols=['cc', 'hospcode'])

# Optionally weeding out hospital 15, which has garbage CCs
not_hosp15 = np.where(records['hospcode'] != 15)[0]
records = records.iloc[not_hosp15, :]

# Prepping the chief complaint column
text = records['cc']

# Vectorizing the text and getting the corpus vocab
text = ['<' + doc + '.' for doc in text]
text_vec = CountVectorizer(binary=False,
                           ngram_range=(1, 1),
                           analyzer='char',
                           decode_error='ignore')
text_vec.fit(text)
vocab = text_vec.vocabulary_
vocab_size = len(list(vocab.keys()))

# Adding 1 to each vocab index to allow for 0 masking
for word in vocab:
    vocab[word] += 1

# Writing the vocabulary to disk
vocab_df = pd.DataFrame.from_dict(vocab, orient='index')
vocab_df['char'] = vocab_df.index
vocab_df.columns = ['value', 'char']
vocab_df.to_csv('data/char_dict.csv', index=False)

# Remembering the value for the end-of-sequence token
eos_val = vocab['.']

# Converting the text strings to sequences of integers
docs = doc_to_char(text)
doc_lengths = np.array([len(doc) for doc in docs])
text_140 = docs[np.where(doc_lengths <= 140)[0]]
max_length = np.max([len(doc) for doc in text_140])
int_sents = np.array([pad_integers(to_integer(doc[:-1], vocab),
                                   max_length, 0) for doc in text_140],
                     dtype=np.uint16)
targets = np.array([pad_integers(to_integer(doc[1:], vocab),
                                   max_length, 0) for doc in text_140],
                     dtype=np.uint16)
n = len(int_sents)

# Setting up the train-test splits
train_indices, test_indices = train_test_split(range(n),
                                               random_state=10221983)

# Writing everything to hdf5 files
output = h5py.File('data/char_sents.hdf5', mode='w')
output['X_train'] = int_sents[train_indices]
output['y_train'] = targets[train_indices]
output['X_test'] = int_sents[test_indices]
output['y_test'] = targets[test_indices]
output.close()
