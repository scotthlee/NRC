'''
Converts text strings to integer sequences,
mostly for use with NLMs and other models with RNNs
'''
import pandas as pd
import numpy as np
import h5py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz, save_npz

import tools.text as tt
import tools.generic as tg

# Importing the text files
records = pd.read_csv(EHR_CSV_FILE)

# Prepping the chief complaint column
text = records[TEXT_COL_NAME]
text = ['start_string ' + doc + ' end_string' for doc in text]
text_series = pd.Series(text)

# First-pass vectorization to get the overall vocab
text_vec = CountVectorizer(binary=False,
                           ngram_range=(1, 1),
                           token_pattern="(?u)\\b\\w+\\b",
                           decode_error='ignore')
text_vec.fit(text)
vocab = text_vec.vocabulary_
vocab_size = len(list(vocab.keys()))

# Adding 1 to each vocab index to allow for 0 masking
for word in vocab:
    vocab[word] += 1

# Writing the vocabulary to disk
vocab_df = pd.DataFrame.from_dict(vocab, orient='index')
vocab_df['word'] = vocab_df.index
vocab_df.columns = ['value', 'word']
vocab_df.to_csv('data/word_dict.csv', index=False)

# Remembering the value for the end-of-sequence token
eos_val = vocab['end_string']

# Converting the text strings to sequences of integers
max_length = 18
doc_lengths = np.array([len(doc.split()) for doc in ten_text])
clip_where = np.where(doc_lengths <= max_length)[0]
clipped_docs = ten_text.iloc[clip_where]

# Weeding out docs with tokens that CountVectorizer doesn't recognize;
# this shouldn't be necessary, but I can't figure out how to debug it.
in_vocab = np.where([np.all([word in vocab.keys()
                             for word in doc.split()])
                     for doc in clipped_docs])
good_docs = clipped_docs.iloc[in_vocab]

# Setting up the train-test splits
n = good_docs.shape[0]
train_indices, test_indices = train_test_split(range(n),
                                               random_state=1234)

# Preparing the HDF5 file to hold the output
output = h5py.File('word_sents.hdf5', mode='w')

# Running and saving the splits for the inputs; going with np.uin16
# for the dtype since the vocab size is much smaller than before
int_sents = np.array([tt.pad_integers(tt.to_integer(doc.split()[:-1], vocab),
                                      max_length, 0) for doc in good_docs],
                     dtype=np.uint16)
output['X_train'] = int_sents[train_indices]
output['X_test'] = int_sents[test_indices]

# And doing the same for the outputs
targets = np.array([tt.pad_integers(tt.to_integer(doc.split()[1:], vocab),
                                    max_length, 0) for doc in good_docs],
                     dtype=np.uint16)
output['y_train'] = targets[train_indices]
output['y_test'] = targets[test_indices]

# Shutting down the HDF5 file
output.close()
