import pandas as pd
import numpy as np

import tools.generic as tg
import tools.text as tt

# Generates a starter array for seeding a RNN-based language model
def make_starter(max_length, indices=None, one_hot=False, vocab_size=None):
    out = np.zeros([1, max_length], dtype=np.uint32)
    if type(indices) != type(None):
        for i, index in enumerate(indices):
            out[0][i] = index
    if one_hot:
        out = np.array([tg.one_hot(num, vocab_size).transpose() for num in out])
    return out
