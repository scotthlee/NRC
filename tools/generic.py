import numpy as np
import pandas as pd

from sys import getsizeof
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

# Gets the number of batches given a batch size and a dataset
def n_batches(n, batch_size):
    return int((n / batch_size) + (n % batch_size))

# Generator function for use with the NRC model;
# Pulled from StackOverflow because I'm lazy.
def nrc_generator(recs, sents, y, batch_size=256, vocab_size=39):
    n = len(y)
    n_batches = int((n / batch_size) + (n % batch_size))
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        rec_batch = recs[index_batch].toarray().astype(dtype=np.uint8)
        sent_batch = sents[index_batch].astype(dtype=np.uint32)
        y_batch = np.array([one_hot(sent, vocab_size).transpose() 
                            for sent in y[index_batch]], dtype=np.uint8)
        counter += 1
        yield([rec_batch, sent_batch], y_batch)
        if (counter < n_batches):
            np.random.shuffle(shuffle_index)
            counter=0

# Simple function for making one-hot vectors
def one_hot(indices, vocab_size, sparse=False, dtype=np.uint8):
    mat = np.zeros((vocab_size, indices.shape[0]), dtype=dtype)
    mat[indices, np.arange(mat.shape[1])] = 1
    mat[0, :] = 0
    return mat

# Converts a matrix of 1-hot vectors or probs to a single dense vector
def densify(mat, axis=2, flatten=False):
    out = np.argmax(mat, axis=axis)
    if flatten:
        out = out.flatten()
    return out

# Runs basic diagnostic stats on binary predictions
def diagnostics(true, pred, average='weighted'):
    sens = recall_score(true, pred, average=average)
    ppv = precision_score(true, pred, average=average)
    f1 = f1_score(true, pred, average=average)
    acc = accuracy_score(true, pred)
    out = pd.DataFrame([sens, ppv, f1, acc]).transpose()
    out.columns = ['sens', 'ppv', 'f1', 'acc']
    return out
