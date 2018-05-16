import numpy as np
import pandas as pd
import h5py

from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import train_test_split
from keras.models import load_model

import tools.generic as tg
import tools.keras as tk
import tools.text as tt
from models.NRC import NRC

# Setting the seed for any train-test splitting
seed = 1234

# Importing and splitting the sparsified syndromic records
records = load_npz(SPARSE_RECORDS_NPZ_FILE)
train_indices, test_indices = train_test_split(range(records.shape[0]),
                                               random_state=seed)
train_recs = records[train_indices]
test_recs = records[test_indices]

# Importing the pretrained autoencoder
ae = load_model(AE_ENCODER_HDF5_FILE)
ae_encoder = ae.layers[1]

# Importing the text files
sents = h5py.File(SENTENCE_INTEGERS_HDF5_FILE, mode='r')
train_sents = sents['X_train'].__array__()
y_train = sents['y_train'].__array__()
test_sents = sents['X_test'].__array__()
y_test = sents['y_test'].__array__()

# Importing the character lookup dictionary
vocab_df = pd.read_csv(WORD_DICTIONARY)
vocab = dict(zip(vocab_df['word'], vocab_df['value']))
eos_val = vocab['end_string']

# Setting some parameters
n = records.shape[0]
sparse_size = records.shape[1]
hidden_size = 128
embedding_size = 128
vocab_size = len(vocab.keys()) + 1
max_length = train_sents.shape[1]

# Setting up the NRC model
nrc = NRC(embedding_size=embedding_size,
          sparse_size=sparse_size,
          hidden_size=hidden_size,
          vocab_size=vocab_size,
          max_length=max_length)

# Loading the training model and building the inference model
nrc.load_training_model('nrc_training.hdf5')

# Running some test captions
nrc.caption(test_recs[0], vocab, method='beam')
