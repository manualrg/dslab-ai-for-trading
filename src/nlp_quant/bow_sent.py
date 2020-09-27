
import os
import pandas as pd

import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()

    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)

    return text


def lemmatize_words(wlm: WordNetLemmatizer, words):
    """
    Lemmatize words

    Parameters
    ----------
    words : list of str
        List of words

    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """

    # Instanciate WordNet Lemmatizer

    # Apply lemmatization supposing verb PoS
    lemmatized_words = [wlm.lemmatize(token, pos='v') for token in words]
    return lemmatized_words

sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']

from src.load_data import  io_utils



def get_sentiment_loughran_mcdonald():
    """
    https://sraf.nd.edu/textual-analysis/resources/
    :return:
    """
    path_loughran_mcdonald = os.path.join(io_utils.raw_path, 'financial_sentiment', '')
    sentiment_df = pd.read_csv(path_loughran_mcdonald + 'loughran_mcdonald_master_dic_2016.csv')

    # Lowercase columns, Remove unused information
    sentiment_df.columns = [column.lower() for column in sentiment_df.columns]

    sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']
    sentiment_df = sentiment_df[sentiments + ['word']]
    sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
    sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

    return sentiment_df