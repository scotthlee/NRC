import pandas as pd
import numpy as np

from scipy.spatial.distance import cosine

def ngrams(sent, max_n=4, flatten=True):
    if type(sent) == type('a'):
        sent = sent.split()
    out = []
    for n in range(1, max_n + 1):
        out.append([' '.join(sent[i:i+n]) for i in range(len(sent) - n+1)])
    if flatten:
        out = [item for items in out for item in items]
    return out

def cider(ref, test, vecs=None, n=4):
    '''
    Calculates the CIDEr metric for a single test sentence relative to a
    single reference sentence. If a vectorizer is provided, the function
    assumes the sentences are in text (rather than vector) format and will
    vectorize them before calculating the statistic.
    
    Arguments:
        1. test = either a string or Numpy array
        2. refs = a Numpy array
        3. vec = a sklearn TfidfVectorizer object
    
    Note that if less than 2 vectorizers are supplied, then the metric will
    need to be calculated for each n-gram level intended to be used and then
    averaged to obtain the final score. Also, this implementation is slow.
    '''
    if type(vecs) != list:
        vecs = [vecs]
    max_length = np.min([len(test.split()), len(ref.split())])
    to_use = np.intersect1d(range(max_length), range(n))
    tests = [vecs[i].transform([test]).toarray() for i in to_use]
    refs = [vecs[i].transform([ref]).toarray() for i in to_use]
    stat = np.mean([1 - cosine(tests[i], refs[i]) for i in to_use])
    return stat

def embedding_similarity(ref, test, vocab, method='average', idf=None):
    '''
    Calculates the CIDEr metric using dense vector representations
    of words as input instead of tf-idf vectors.
    
    Arguments:
        1. refs = the candidate sentence in string format
        2. test = the reference sentence(s) in string format
        3. vocab = a lookup dictionary for the embeddings
        4. method = whether to 'sum' the embeddings or 'average' them before
            calculating cosine similarity between candidate and reference vecs
    '''
    test_split = [word for word in test.split() if word in list(vocab.keys())]
    ref_split = [word for word in ref.split() if word in list(vocab.keys())]
    test_vecs = [vocab[term] for term in test_split]
    ref_vecs = [vocab[term] for term in ref_split]
    
    # Adding the word vectors to get the phrase embedding
    test_embedding = np.sum(test_vecs, axis=0)
    ref_embedding = np.sum(ref_vecs, axis=0)
    
    # Averaging the vectors
    if method == 'average':
        test_embedding /= len(test_split)
        ref_embedding /= len(ref_split)
    
    # Computing the statistic
    stat = 1 - cosine(test_embedding, ref_embedding)
    return stat

def violet(ref, test, n=4, clip=True):
    ''''
    Calculates Recall-Oriented Understudy for Gisting Evaluation, or ROUGE, as
    well as Bilingual Evaluation Understudy, or BLEU, for a candidate sentence
    based on a single reference sentence. Also calculates F1 based on the two
    previous statistics, i.e. by obtaining their harmonic mean.
    
    Arguments:
        1. ref: the reference sentence as as single string
        2. test: the candidate  sentence as a single string
        3. n: the maximum number of n-grams to consider
        4. clip: whether to limit n to the length of the shortest sentence
    '''
    test = test.split()
    ref = ref.split()
    if clip:
        n = np.min([len(test), len(ref), n])
    test_grams = np.unique(ngrams(test, n))
    ref_grams = np.unique(ngrams(ref, n))
    bleu = np.sum([gram in ref_grams for gram in test_grams]) / len(test_grams)
    rouge = np.sum([gram in test_grams for gram in ref_grams]) / len(ref_grams)
    if bleu + rouge == 0:
        violet = 0
    else:
        violet = 2 * (bleu * rouge) / (bleu + rouge)
    return pd.DataFrame(np.array([bleu, rouge, violet]).reshape(1, -1),
                        columns=['bleu', 'rouge', 'violet'])
