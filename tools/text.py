'''Objects and methods to support text corpus storage and manipulation'''
import numpy as np
import pandas as pd
import re
import string

from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer

# Converts a string corpus to unicode. Kinda redundant.
def to_unicode(corpus, encoding='utf-8', errors='replace'):
    utf_corpus = [str(text, encoding=encoding, errors=errors)
                  for text in corpus]
    return utf_corpus

# Converts a unicode corpus to string
def to_string(corpus, errors='ignore'):
    str_corpus = [doc.encode('ascii', errors) for doc in corpus]
    return str_corpus

# Converts a single unicode string to a list of characters
def word_to_char(word, output_type='array'):
    if output_type == 'array':
        return np.array([char for char in word])
    elif output_type == 'list':
        return [char for char in word]

# Converts a list of words to an array of characters
def doc_to_char(doc, concatenate=False):
    if concatenate:
        return np.concatenate(np.array([word_to_char(word) for word in doc]))
    else:
        return np.array([word_to_char(word) for word in doc])

# Gets the frequency distribution of terms in a corpus
def freq_dist(corpus, vocab=None):
    if vocab is None:
        vocab = new_vectorizer().fit(corpus).vocabulary_
    dist = dict(list(zip(list(vocab.keys()), 
                         np.zeros(len(vocab), 
                                  dtype=int))))
    for doc in corpus:
        for word in doc:
            dist[word] += 1
    return dist

# Does all manner of preprocessing on an a tokenized corpus
def remove_rare(doc, freqs, min_count=5, replacement='unk'):
	i = 0
	for word in doc:
		if freqs[word] < min_count:
			doc[i] = replacement
		i += 1
	return doc

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

# Pads an 2D array of (column-vector) word embeddings
def pad_embeddings(phrase, max_length, dtype='float32'):
	if len(phrase.shape) == 1:
		phrase = phrase.reshape([phrase.shape[0], 1])
	pad_size = max_length - phrase.shape[1]
	embedding_size = phrase.shape[0]
	return np.concatenate((phrase, np.zeros([embedding_size, pad_size],
                                         dtype=dtype)), axis=1)

# Converts a corpus to an array of skipgrams
# with shape [num_grams, 2 + k], where k
# is the number of negative samples and num_grams = 2 * win_size * num_wins
def make_skipgrams(corpus, winsize=2, k=5, flatten=True):
    num_win = len(list(range(winsize, len(corpus) - winsize)))
    num_grams = 2 * winsize * num_win
    win_width = 2 * winsize + 1
    out = np.zeros([num_win, win_width - 1, 2 + k], dtype=int)
    for i in range(num_win):
        window = corpus[i:i+win_width]
        center = window[winsize]
        del(window[winsize])
        for j in range(len(window)):
            out[i, j, 0] = center
            out[i, j, 1] = window[j]
            out[i, j, 2:] = np.random.randint(0, vocabulary_size, size=k)
    if flatten:
        out = out.reshape([out.shape[0] * (2 * winsize), out.shape[2]])
    return out

# Kludge to prevent isalph() from checking for non-ASCII characters
def is_alpha(char):
    return char in string.ascii_lowercase

# Functions for removing specific types of strings
def remove_special_old(docs):
    docs = [doc.replace('\\', ' ') for doc in docs]
    docs = [doc.replace('_', ' ') for doc in docs]
    docs = [doc.replace('"', ' ') for doc in docs]
    docs = [doc.replace('^', ' ') for doc in docs]
    docs = [doc.replace('`', ' ') for doc in docs]
    docs = [doc.replace('.', ' ') for doc in docs]
    docs = [doc.replace(',', ' ') for doc in docs]
    docs = [doc.replace('|', ' ') for doc in docs]
    docs = [doc.replace("'", '') for doc in docs]
    docs = [doc.replace('/', ' ') for doc in docs]
    docs = [doc.replace('?', ' ') for doc in docs]
    docs = [doc.replace('(', ' ') for doc in docs]
    docs = [doc.replace(')', ' ') for doc in docs]
    docs = [doc.replace('*', ' ') for doc in docs]
    docs = [doc.replace('!', ' ') for doc in docs]
    docs = [doc.replace('{', ' ') for doc in docs]
    docs = [doc.replace('}', ' ') for doc in docs]
    docs = [doc.replace('[', ' ') for doc in docs]
    docs = [doc.replace(']', ' ') for doc in docs]
    docs = [doc.replace('+', ' ') for doc in docs]
    docs = [doc.replace('&', ' ') for doc in docs]
    docs = [doc.replace('-', ' ') for doc in docs]
    docs = [doc.replace('@', ' ') for doc in docs]
    docs = [doc.replace('$', ' ') for  doc in docs]
    docs = [doc.replace(':', ' ') for doc in docs]
    docs = [doc.replace(';', ' ') for doc in docs]
    docs = [doc.replace('#', ' ') for doc in docs]
    docs = [doc.replace('~', ' ') for doc in docs]
    docs = [doc.replace('=', ' equals ') for doc in docs]
    docs = [doc.replace('%', ' percent ') for doc in docs]
    return docs

def remove_special(docs):
    docs = [doc.lower() for doc in docs]
    docs = [[char for char in doc if is_alpha(char)
             or char.isnumeric() or char == ' ']
            for doc in docs]
    docs = [''.join(doc) for doc in docs]
    return docs

# Converts numerals to text, i.e. 1 to 'one'
def remove_numerals(docs):
    docs = [doc.replace('0', ' zero ') for doc in docs]
    docs = [doc.replace('1', ' one ') for doc in docs]
    docs = [doc.replace('2', ' two ') for doc in docs]
    docs = [doc.replace('3', ' three ') for doc in docs]
    docs = [doc.replace('4', ' four ') for doc in docs]
    docs = [doc.replace('5', ' five ') for doc in docs]
    docs = [doc.replace('6', ' six ') for doc in docs]
    docs = [doc.replace('7', ' seven ') for doc in docs]
    docs = [doc.replace('8', ' eight ') for doc in docs]
    docs = [doc.replace('9', ' nine ') for doc in docs]
    return docs

# Removes extra whitespace for a list of text strings. Kludgy, but useful.
def remove_whitespace(docs):
	docs = [' '.join(doc.split()) for doc in docs]
	docs = [doc.rstrip() for doc in docs]
	docs = [doc.lstrip() for doc in docs]
	return docs

# Removes 'nan' strings for a list of strings
def remove_nan(docs):
    docs = [doc.replace('nan', '') for doc in docs]
    return docs

# Helper function for reading and cleaning the CCS
# dictionaries downloaded directly from HCUP
def read_ccs(ccs_filepath):
    infile = pd.read_csv(ccs_filepath)
    ccs = infile.iloc[:, [0, 1, 2, 3]]
    ccs.columns = ['icd_code', 'ccs_code', 'icd_descr', 'ccs_descr']
    ccs['icd_code'] = [code.lower() for code in ccs['icd_code']]
    return ccs

# Function that cleans up a free-text column
def clean_column(docs, remove_empty=True, numerals=True):
    docs = [doc.lower() for doc in docs]
    if remove_empty:
        docs = docs[np.where(docs != 'nan')]
    else:
        docs = [doc.replace('nan', '') for doc in docs]
    docs = remove_special(docs)
    if numerals:
        docs = remove_numerals(docs)
    docs = [doc.replace(' ve ', ' have ') for doc in docs]
    docs = [doc.replace(' c o ', ' co ') for doc in docs]
    docs = [doc.replace(' b l ', ' bl ') for doc in docs]
    docs = [doc.replace(' s p ', ' sp ') for doc in docs]
    docs = [doc.replace(' pn ', ' pain ') for doc in docs]
    docs = [doc.replace(' n v d ', ' nvd ') for doc in docs]
    docs = [doc.replace('cc', ' ') for doc in docs]
    docs = [doc.replace('chief complaint quote', ' ') for doc in docs]
    notes = [doc.replace('x zero d', ' ') for doc in docs]
    notes = [doc.replace('zero a', ' ') for doc in docs]
    docs = [re.sub(r'\@.*\@', ' ', doc) for doc in docs]
    docs = remove_whitespace(docs)
    return docs

# Cleaning up the CCS code column; should probably be done somewhere else
def clean_ccs_column(col):
	clean = [visit.replace('[', '') for visit in col]
	clean = [visit.replace(']', '') for visit in clean]
	clean = [visit.replace(',', '') for visit in clean]
	clean = [keep_unique(visit) for visit in clean]
	clean = [visit.replace('None', '') for visit in clean]
	return np.array(clean)
