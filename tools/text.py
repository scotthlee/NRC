'''Objects and methods to support text corpus storage and manipulation'''
import numpy as np
import pandas as pd
import re
import string

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer


# Looks up a dict key up by its values
def get_key(value, dic, add_1=False, pad=0):
    if add_1 and value != pad:
        value += 1
    if value == pad:
        out = 'pad'
    else:
        out = list(dic.keys())[list(dic.values()).index(value)]
    return out

# Uses get_key to lookup a sequence of words or characters
def ints_to_text(values, dic, level='word', remove_bookends=True):
    good_values = values[np.where(values != 0)[0]]
    if remove_bookends:
        good_values = good_values[1:-1]
    tokens = [get_key(val, dic) for val in good_values]
    if level == 'word':
        return ' '.join(tokens)
    return ''.join(tokens)

# Converts a text-type column of a categorical variable to integers
def text_to_int(col):
    vec = CountVectorizer(token_pattern="(?u)\\b\\w+\\b")
    vec.fit(col)
    vocab = vec.vocabulary_
    dict_values = [vocab.get(code) for code in col]
    return {'values':dict_values, 'vocab':vocab}

# Converts a list of tokens to an array of integers
def to_integer(tokens, vocab_dict, encode=False,
               subtract_1=False, dtype=np.uint32):
    if encode:
        tokens = [str(word, errors='ignore') for word in tokens]
    out = np.array([vocab_dict.get(token) for token in tokens], dtype=dtype)
    if subtract_1:
        out = out - 1
    return out

# Pads a 1D sequence of integers (representing words)
def pad_integers(phrase, max_length, padding=0):
	pad_size = max_length - len(phrase)
	return np.concatenate((phrase, np.repeat(padding, pad_size)))
