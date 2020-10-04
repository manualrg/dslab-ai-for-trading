import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer


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

    # Apply lemmatization supposing verb PoS
    lemmatized_words = [wlm.lemmatize(token, pos='v') for token in words]
    return lemmatized_words


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag of words from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """

    # CountVectorizer implements the following steps by default:
    # strip_accents: None (Not considered)
    # lowercase: True (sentiment words are lowercased)
    # stop_words: None (not necessary, as vocabulary is supplied)
    count_vect = CountVectorizer(vocabulary=sentiment_words).fit(docs)
    bag_of_words = count_vect.transform(docs)
    return bag_of_words.toarray()


def get_tfidf(sentiment_words, docs):
    """
    Generate TFIDF values from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """

    tfidf_vect = TfidfVectorizer(vocabulary=sentiment_words).fit(docs)
    tfidf = tfidf_vect.transform(docs)
    return tfidf.toarray()


def get_jaccard_similarity(bow_matrix):
    """
    Get jaccard similarities for neighboring documents

    Parameters
    ----------
    bow_matrix : 2-d array-like (Numpy Ndarray or Pandas DataFrame) of int
        Bag of words representation for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    jaccard_similarities : list of float
        Jaccard similarities for neighboring documents
    """
    if isinstance(bow_matrix, pd.DataFrame):
        matrix = bow_matrix.values
    else:
        matrix = bow_matrix

    assert np.issubclass_(matrix.dtype.type, np.integer), "Input bow_matrix should be integer dtype"

    jaccard_similarities = []
    n_rows, _ = matrix.shape
    matrix = np.where(matrix > 0, True, False)
    for row_idx in range(n_rows - 1):
        # Fetch vector rows from matrix in 1d shape
        vec_t = matrix[row_idx, :]
        vec_t1 = matrix[row_idx + 1, :]
        # Compute distance
        dist = jaccard_score(y_true=vec_t, y_pred=vec_t1)
        # Append to al list where element "i" is d(v[t,:], v[t+1,:])
        jaccard_similarities.append(dist)
    return jaccard_similarities


def get_cosine_similarity(bow_matrix):
    """
    Get cosine similarities for each neighboring TFIDF vector/document

    Parameters
    ----------
    bow_matrix : 2-d array-like (Numpy Ndarray or Pandas DataFrame) of float or nint
        TFIDF or BoW representation for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    cosine_similarities : list of float
        Cosine similarities for neighboring documents
    """
    if isinstance(bow_matrix, pd.DataFrame):
        matrix = bow_matrix.values
    else:
        matrix = bow_matrix

    cosine_similarities = []
    n_rows, _ = matrix.shape

    for row_idx in range(n_rows - 1):
        # Fetch vector rows from matrix in 2d shape: (1, n_words)
        vec_t = matrix[row_idx].reshape(1, -1)
        vec_t1 = matrix[row_idx + 1].reshape(1, -1)
        # Append to a list where element "i" is d(v[t,:], v[t+1,:])
        dist = cosine_similarity(vec_t, vec_t1).reshape(-1)[0]
        # Compute distance
        cosine_similarities.append(dist)

    return cosine_similarities