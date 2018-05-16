import numpy as np
import pandas as pd
import h5py

from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

# Setting some global parameters
n = records.shape[0]
sparse_size = records.shape[1]
hidden_size = 128
embedding_size = 128
vocab_size = len(vocab.keys()) + 1
max_length = train_sents.shape[1]
batch_size = 512
epochs = 50
n_train = len(y_train)
n_test = len(y_test)
train_steps = tg.n_batches(n_train, batch_size)
test_steps = tg.n_batches(n_test, batch_size)

# Setting up the data generators
train_gen = tg.nrc_generator(train_recs, train_sents, y_train,
                             vocab_size=vocab_size,
                             batch_size=batch_size)
test_gen = tg.nrc_generator(test_recs, test_sents, y_test,
                            vocab_size=vocab_size,
                            batch_size=batch_size)

# Making some callbacks
checkpointer = ModelCheckpoint(filepath='nrc_training.hdf5',
                               save_best_only=True,
                               verbose=1)
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=2)

# Setting up the NRC model
nrc = NRC(embedding_size=embedding_size,
          sparse_size=sparse_size,
          hidden_size=hidden_size,
          vocab_size=vocab_size,
          max_length=max_length)

# Building and running the model in training mode
nrc.build_training_model(trainable_records=True,
                         encoding_layer=ae_encoder)
nrc.training_model.compile(optimizer='adam',
                           loss='categorical_crossentropy')

nrc.training_model.fit_generator(train_gen, verbose=1,
                                 steps_per_epoch=train_steps,
                                 validation_data=test_gen,
                                 validation_steps=test_steps,
                                 epochs=epochs,
                                 callbacks=[checkpointer, early_stopping])

# Testing the model in inference mode
nrc.build_inference_model()
nrc.caption(test_recs[0], vocab)
