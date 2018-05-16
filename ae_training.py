import pandas as pd
import numpy as np

from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from models.unsupervised import autoencoder

# Quick function for thresholding probabilities
def threshold(probs):
    return (probs > .5).astype(int)

# Making some callbacks
checkpointer = ModelCheckpoint(filepath='ae_model.hdf5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Importing the data
records = load_npz(SPARSE_RECORDS_NPZ)
X_train, X_test = train_test_split(records, random_state=10221983)

# Setting some global parameters
sparse_dim = records.shape[1]
embedding_dim = 128

# Training the model and loading the one with the lowest validation loss
mod = autoencoder(sparse_dim, embedding_dim)
mod.compile(optimizer='adam', loss='binary_crossentropy')
mod.fit(X_train, X_train,
                epochs=100,
                batch_size=1024,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[checkpointer, early_stopping])
