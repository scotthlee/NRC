import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, save_npz, csr_matrix

from tools.generic import gigs, one_hot
from tools.text import remove_nan, remove_special

# Just a few functions to support making a sparse array
def to_sparse(col, dtype='str', vocab_size=None, vocab_only=False):
    if dtype=='str':
        vec = CountVectorizer(binary=True,
                              ngram_range=(1, 1),
                              token_pattern="(?u)\\b\\w+\\b")
        data = vec.fit_transform(col)
        vocab = sorted(vec.vocabulary_.keys())
        if vocab_only:
            return vocab
    else:
        data = csr_matrix(one_hot(col, vocab_size)).transpose()
        vocab = np.unique(col)
    return {'data':data, 'vocab':vocab}

# Reading in the data
slim_cols = [COLUMNS_TO_USE]
records = pd.read_csv('~/data/syndromic/good_cc_records.csv',
                      usecols=slim_cols)

# Making the sparse matrices
sparse_out = [to_sparse(records[col].astype(str)) for col in slim_cols]
sparse_csr = hstack([col['data'] for col in sparse_out], format='csr')
sparse_vocab = [col['vocab'] for col in sparse_out]
sparse_vocab = pd.Series([item for sublist in sparse_vocab
                          for item in sublist])

# Writing the files to disk
save_npz('sparse_records', sparse_csr)
sparse_vocab.to_csv('sparse_vocab.csv', index=False)
