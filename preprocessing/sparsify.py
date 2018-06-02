import pandas as pd
import numpy as np

from tools.generic import gigs, one_hot, to_sparse

# Reading in the data
slim_cols = [COLUMNS_TO_USE]
records = pd.read_csv(EHR_CSV_FILE, usecols=slim_cols)

# Making the sparse matrices
sparse_out = [to_sparse(records[col].astype(str)) for col in slim_cols]
sparse_csr = hstack([col['data'] for col in sparse_out], format='csr')
sparse_vocab = [col['vocab'] for col in sparse_out]
sparse_vocab = pd.Series([item for sublist in sparse_vocab
                          for item in sublist])

# Writing the files to disk
save_npz('sparse_records.npz', sparse_csr)
sparse_vocab.to_csv('sparse_vocab.csv', index=False)
