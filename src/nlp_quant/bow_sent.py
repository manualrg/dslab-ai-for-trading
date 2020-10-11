import os
import gzip
import logging
import datetime as dt

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import pickle

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm


# Gets or creates a logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.INFO)


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


def get_tfidf(sentiment_words, docs, flg_sparse=True):
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
    if flg_sparse:
        return tfidf
    else:
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


def filenames_to_index(in_listdir):

    idx_df = pd.DataFrame(data=[doc.split(".")[0].split("_") for doc in in_listdir], columns=['ticker', 'doc_type', 'date'])
    idx_df['date'] = idx_df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
    doc_meta = pd.MultiIndex.from_frame(idx_df[['ticker', 'date']])

    return doc_meta



def batch_tfidf(inpath, batch_size, lemmatizer, stopwords, re_word_pattern, vocabs):
    in_listdir = os.listdir(inpath)

    n_batches = int(len(in_listdir)/batch_size)
    in_listdir_batches = np.array_split(in_listdir, n_batches)
    sent_tfidf_dict = dict(zip(vocabs.keys(), [[]]*len(vocabs)))

    for batch in tqdm(in_listdir_batches, desc=f'Extracting tf-idf', unit='batch'):
        tickers_lst = []
        docs_meta = filenames_to_index(batch)
        docs_lst = []
        # Read docs and create a list of documents to process: docs_lst
        for file in batch:
            ticker, doc_type, date = file.split("_")
            tickers_lst.append(ticker)
            date = date.split(".")[0]
            infilename = inpath + file

            with gzip.open(infilename, "rb") as f:
                doc = f.read()
            doc = doc.decode()
            docs_lst.append(doc)
        # Compute tf-idf on a batch of documents
        # a dict {sentiment: pandas DF} is returned. Each pandas DF is a tf-idf represenstantion of a batch of documents
        # indexed by docs_meta as ticker-date pairs
        # columns are each sentiment specific vocabulary
        # TODO: Add doc len
        tfidf_batch = nlp_pipeline(docs=docs_lst, docs_meta=docs_meta,
                     lemmatizer=lemmatizer, stopwords=stopwords, re_word_pattern=re_word_pattern, vocabs=vocabs)
        # Add pandas DFs to an inner list {sentiment: [tfidf_batch (pandas DF)]}
        for sent_key, sent_vocab in vocabs.items():
            sent_tfidf_dict[sent_key] = sent_tfidf_dict[sent_key] + [tfidf_batch[sent_key]]

        # logging set of tickers
        logger.info(f'Tickers in batch: {list(set(tickers_lst))}')
    # Concatenate inner lists elements {sentiment: tfidf_app}
    for sent_key, sent_vocab in vocabs.items():
        sent_tfidf_dict[sent_key] = pd.concat(sent_tfidf_dict[sent_key], axis=0)

    return sent_tfidf_dict

def nlp_pipeline(docs, docs_meta, lemmatizer, stopwords, re_word_pattern, vocabs):

    assert len(docs) == len(docs_meta), "docs and docs_meta have the same length"

    docs_lst = []
    for doc in docs:
        # tokenize
        doc_lemma = lemmatize_words(lemmatizer, re_word_pattern.findall(doc))
        # Remove stopwords
        doc_lemma = " ".join([word for word in doc_lemma if word not in stopwords])
        # TODO: Add doc len
        docs_lst.append(doc_lemma)

    sent_tfidf_dict = {}
    for sent_key, sent_vocab in vocabs.items():
        sent_tfidf = pd.DataFrame.sparse.from_spmatrix(index=docs_meta,
                                  data=get_tfidf(sentiment_words=sent_vocab, docs=docs_lst),
                                  columns=sent_vocab)
        sent_tfidf_dict[sent_key] = sent_tfidf

    return sent_tfidf_dict


def write_sent_tfidf_dict(path, name, sent_tfidf_dict):
    with open(path + name, 'wb') as file:
        pickle.dump(sent_tfidf_dict, file)


def read_sent_tfidf_dict(path, name):
    with open(path + name, 'rb') as file:
        sent_tfidf_dict = pickle.load(file)

    return sent_tfidf_dict


def batch_doc_len(inpath, batch_size, re_word_pattern):
    in_listdir = os.listdir(inpath)

    n_batches = int(len(in_listdir) / batch_size)
    in_listdir_batches = np.array_split(in_listdir, n_batches)

    doc_len_df_lst = []
    for batch in tqdm(in_listdir_batches, desc=f'Extracting tf-idf', unit='batch'):
        docs_meta = filenames_to_index(batch)
        docs_len_lst = []
        # Read docs and create a list of documents to process: docs_lst
        for file in batch:
            ticker, doc_type, date = file.split("_")
            infilename = inpath + file

            with gzip.open(infilename, "rb") as f:
                doc = f.read()
            doc = doc.decode()
            docs_len_lst.append(len(re_word_pattern.findall(doc)))  # Compute doc length

        doc_len_df_lst.append(pd.Series(index=docs_meta, data=docs_len_lst, name='doc_len'))

    return pd.concat(doc_len_df_lst)


def compute_sentiment_alpha_factor(sent_scores, group_columns, sector_col, score_col):
    sent_scores_cp = sent_scores.copy()
    # sector de-mean
    sent_scores_cp['sector_mean'] = sent_scores.groupby(sector_col)[score_col].transform(np.mean)
    sent_scores_cp['demean'] = sent_scores_cp[score_col] - sent_scores_cp['sector_mean']
    # rank
    sent_scores_cp['ranked'] = sent_scores_cp.groupby(group_columns)['demean'].transform(rankdata)
    # zscore
    sent_scores_cp['mu'] = sent_scores_cp.groupby(group_columns)['ranked'].transform(np.mean)
    sent_scores_cp['sigma'] = sent_scores_cp.groupby(group_columns)['ranked'].transform(np.std)
    sent_alphas = (sent_scores_cp['ranked'] - sent_scores_cp['mu']) / sent_scores_cp['sigma']
    sent_alphas.name = score_col

    return sent_alphas

def get_combined_tfidf(tf_idf_by_sent: dict):
    removed_words = {}

    for i, (sent_key, tfidf_mat) in enumerate(tf_idf_by_sent.items()):
        if i == 0:
            tfidf = tfidf_mat
        else:
            input_cols = tfidf_mat.columns.tolist()
            current_cols = tfidf.columns.tolist()
            remove_dupl = list(set(current_cols).intersection(set(input_cols)))

            if len(remove_dupl) > 0:
                removed_words[sent_key] = remove_dupl
                select_cols = [x for x in input_cols if x not in remove_dupl]

            else:
                select_cols = input_cols
            tfidf = tfidf.join(tfidf_mat[select_cols])

    print('Number of removed words:')
    for sent_key, rem_w in removed_words.items():
        print(f'{sent_key}: {len(rem_w)}')

    return tfidf