from dateutil.relativedelta import relativedelta
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import logging
import spacy

#from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser

from src.nlp_quant.sec_filings import parse_sec_tenks

sec_path = parse_sec_tenks.sec_path
clean_path = parse_sec_tenks.clean_path
ngram_path = parse_sec_tenks.sec_path / 'tenks_ngrams'
stats_path = parse_sec_tenks.sec_path / 'tenks_corpus_stats'

#for path in [ngram_path, stats_path]:
#    if not path.exists():
#        path.mkdir(parents=True)
#        unigrams = ngram_path / 'ngrams_1.txt'

# Parameters
phrases_args = {'min_count': 25,  # ignore terms with a lower count
                'threshold': 0.5,  # accept phrases with higher score
                'max_vocab_size': 40000000,  # prune of less common words to limit memory use
                'delimiter': b'_',  # how to join ngram tokens
                'progress_per': 50000,  # log progress every
                'scoring': 'npmi'}
max_ngram_length = 3

def create_unigrams(min_length=3, text_col='text', inpath=clean_path, outpath_ngram=ngram_path, outpath_stats=stats_path,
                    **kwargs):
    """
    Dumps a corpus consisting of files stored as .csv in clean_path into a file of sentences in unigrams_file
    In input copurs of .csv files, the following columns may exists: item, sentence, text.
    Also a vocabulary is made (stats_section_vocab)
    In addition, and optionally, if item_col is passed as optional kwarg, at each document,
     the number of sentences for each item value will be counted (stats_select_sents)
    :param min_length: minimum length sentence to be included in texts
    :param kwargs:
        item_col: column in input pandas DataFrame
        sections: a list of strings of targeted sections, applies a filter to item_col
    """
    item_col = kwargs.get("item_col", None)
    sections = kwargs.get("sections", None)
    if not outpath_ngram.exists():
        outpath_ngram.mkdir(exist_ok=True, parents=True)
    if not outpath_stats.exists():
        outpath_stats.mkdir(exist_ok=True, parents=True)

    unigrams_file = outpath_ngram / 'ngrams_1.txt'
    stats_select_sents = outpath_stats / 'selected_sentences.csv'
    stats_section_vocab = outpath_stats / 'sections_vocab.csv'

    texts = []

    sentence_counter = Counter()
    vocab = Counter()
    for i, f in enumerate(inpath.glob('*.csv')):
        if i % 1000 == 0:
            print(i, end=' ', flush=True)
        df = pd.read_csv(f)

        if item_col:
            df[item_col] = df[item_col].astype(str)
            if sections:
                df = df[item_col].isin(sections)
            sentence_counter.update(df.groupby(item_col).size().to_dict())

        for sentence in df[text_col].dropna().str.split().tolist():
            if len(sentence) >= min_length:
                vocab.update(sentence)
                texts.append(' '.join(sentence))

    # persist
    if item_col:
        (pd.DataFrame(sentence_counter.most_common(),
                      columns=['item', 'sentences'])
         .to_csv(stats_select_sents, index=False))

    (pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
     .to_csv(stats_section_vocab, index=False))

    unigrams_file.write_text('\n'.join(texts), encoding='utf-8')  #



def create_ngrams(max_length=3, phrases_args = {'min_count': 25}, ngram_path=ngram_path, stats_path=stats_path):
    """
    Takes a file of sentences from  ngram_path/ngram_1.txt and perform phrases detection, at each pass,
    a file ngram_path/ngram_{}.txt is created, where each line is a senteces and ngrams are joined.
    :param max_length: Max ngrams size (number of passes=max_length-1)
    :param phrases_args: Arguments to pass to gensim.models.phrases.Phrases
    Iteratively creates ngram_path/ngram_{}.txt and append to a pandas DF resulting ngram vocabulary,
    that is finally persisted in parquet in stats_path
    """
    if not ngram_path.exists():
        ngram_path.mkdir(exist_ok=True, parents=True)
    if not stats_path.exists():
        stats_path.mkdir(exist_ok=True, parents=True)


    n_grams_df_lst = []
    start = time()
    for n in range(2, max_length + 1):
        ngrams_in = ngram_path / f'ngrams_{n - 1}.txt'
        ngrams_out = ngram_path / f'ngrams_{n}.txt'
        print(n, end=' ', flush=True)

        sentences = LineSentence(ngrams_in)  # streams sentences from a file of sentences
        phrases = Phrases(sentences=sentences, **phrases_args)  # phrase detection

        n_grams_df = pd.DataFrame([[k.decode('utf-8'), v] for k, v in phrases.export_phrases(sentences)],
                         columns=['phrase', 'score']).assign(length=n)
        n_grams_df_lst.append(n_grams_df)

        # Creates a new file of sentences
        grams = Phraser(phrases)
        sentences = grams[sentences]
        ngrams_out.write_text('\n'.join([' '.join(s) for s in sentences]), encoding='utf-8')

    # Persists
    n_grams = pd.concat(n_grams_df_lst, axis=0)
    n_grams = n_grams.sort_values('score', ascending=False)
    n_grams.phrase = n_grams.phrase.str.replace('_', ' ')
    n_grams['ngram'] = n_grams.phrase.str.replace(' ', '_')

    n_grams.to_parquet(stats_path / 'ngrams.parquet')

    print('\n\tDuration: ', parse_sec_tenks.format_time(time() - start))
    print('\tngrams: {:,d}\n'.format(len(n_grams)))
    print(n_grams.groupby('length').size())