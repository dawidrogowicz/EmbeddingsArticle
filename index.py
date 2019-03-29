import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os
import sys
import re
import nltk
import collections
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from embeddings import train_embeddings, visualize_embeddings
from model import train_model, test_model, predict
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
wnl = WordNetLemmatizer()

# CONSTANTS
STOP_WORDS = set(stopwords.words('english'))
DICTIONARIES_PATH = 'pickles/dictionaries.pickle'
DATA_X_PATH = 'pickles/data_x.pickle'
EMBEDDINGS_PATH = 'pickles/embeddings.pickle'
LEXICON_PATH = 'pickles/lexicon.pickle'
PREDICTION_PATH = 'pickles/prediction.pickle'


# FUNCTIONS
def main(argv=sys.argv):
    if argv[1] == 'rm':
        for arg in argv[2:]:
            del_dir(arg)


def del_dir(dirname):
    for subdir in os.listdir(dirname):
        subpath = os.path.join(dirname, subdir)

        if os.path.isfile(subpath):
            try:
                os.unlink(subpath)
            except Exception as e:
                print('Could not delete file ', subpath)
                print(e)
        else:
            del_dir(subpath)
            os.rmdir(subpath)


def token_valid(token):
    return ((token not in STOP_WORDS)
            and (len(token) > 2)
            and (len(token) < 20)
            and re.match(r'^[a-z]+$', token))


def create_lexicon(line_list, n_tokens=20000):
    _lexicon = list()
    word_list = list()
    for line in tqdm(line_list):
        words = word_tokenize(line.lower())
        words = [wnl.lemmatize(token) for token in words if token_valid(token)]
        word_list.extend(words)

    print('Unique words: ', len(set(_lexicon)))
    # use only n most common words
    _lexicon = ['UNKNOWN'] + [count[0] for count in Counter(word_list).most_common(n_tokens - 1)]
    return _lexicon, word_list


def create_dictionary(_lexicon):
    _dictionary = dict()
    for entry in _lexicon:
        _dictionary[entry] = len(_dictionary)
    _reverse_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return _dictionary, _reverse_dictionary


def convert_sentences(words, _dictionary):
    words = [_dictionary.get(token, 0) for token in tqdm(words)]
    # remove neighbour zeros
    words = [x for i, x in enumerate(tqdm(words)) if sum(words[i - 1:i]) > 0]

    return words


# START

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(main())

with open('./dataset/dataset.txt', 'r') as f:
    data_x = f.readlines()

# LEXICON
# If lexicon file exists, load it
# else, create new lexicon and save it to the file
if os.path.exists(LEXICON_PATH):
    with open(LEXICON_PATH, 'rb') as f:
        lexicon_out = pickle.load(f)
    print('lexicon loaded')
else:
    lexicon_out = create_lexicon(data_x)
    with open(LEXICON_PATH, 'wb') as f:
        pickle.dump(lexicon_out, f)
    print('lexicon created')

lexicon, data_x = lexicon_out

print('lexicon: ', lexicon[:5])
# DICTIONARIES
# If dictionaries file exists, load it
# else, create new dictionaries and save them to the file
if os.path.exists(DICTIONARIES_PATH):
    with open(DICTIONARIES_PATH, 'rb') as f:
        dictionary, reverse_dictionary = pickle.load(f)
    print('dictionaries loaded')
else:
    dictionaries = create_dictionary(lexicon)
    dictionary, reverse_dictionary = dictionaries
    with open(DICTIONARIES_PATH, 'wb') as f:
        pickle.dump(dictionaries, f)
    print('dictionaries created')

# DATA X RAW
# If features file exists, load it
# else, process new features and save it to the file
if os.path.exists(DATA_X_PATH):
    with open(DATA_X_PATH, 'rb') as f:
        data_x = pickle.load(f)
    print('features loaded')
else:
    data_x = convert_sentences(data_x, dictionary)
    with open(DATA_X_PATH, 'wb') as f:
        pickle.dump(data_x, f)
    print('features processed')


# EMBEDDINGS
# If embeddings file exists, load it
# else, create new embeddings and save it to the file
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print('embeddings loaded')
else:
    embeddings = train_embeddings(data_x, len(lexicon))
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print('embeddings processed')

# Embeddings visualisation
visualize_embeddings(lexicon, embeddings)

