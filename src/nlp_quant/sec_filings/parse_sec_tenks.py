from dateutil.relativedelta import relativedelta
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import logging
import spacy

#from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
#from gensim.models.phrases import Phrases, Phraser

from src.load_data import io_utils

#data_path = os.environ.get("DATA_PATH")
#raw_path = os.path.join(data_path, "raw", "")
#interim_path = os.path.join(data_path, "interim", "")
#processed_path = os.path.join(data_path, "processed", "")

logging.basicConfig(
        filename='preprocessing.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')

# Define folder structure
sec_path = Path(io_utils.raw_path, 'sec_filings')  # root: should have /tenks/ and filing_index.csv
filing_path = sec_path / 'tenks'  # parsed .txt sec documents from
sections_path = sec_path / 'tenks_sections'  # split each document in .csv file, where each row is a section
clean_path = sec_path / 'tenks_selected_sections'  # filter sections, clean text and split in sentences
lemma_path = sec_path / 'tenks_lemma_selected_sections'  # filter sections, clean text and lemmatize documents

# Parameters
sections = ['1', '1a', '7', '7a']
max_doc_length = 6000000
min_sentece_length = 5

def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'

def filing_section_identification(doc: str):
    """
    Split each input parsed filing in sections
    :param doc: parsed filing
    :return: items is a pandas Seres name=text, index=item, where store each section string is a value,
    and section name (item) is index
    """
    items = pd.Series(name='text', dtype=str)
    for section in doc.lower().split('°'):
        if section.startswith('item '):
            if len(section.split()) > 1:
                item = section.split()[1].replace('.', '').replace(':', '').replace(',', '')
                text = ' '.join([t for t in section.split()[2:]])
                if items.get(item) is None or len(items.get(item)) < len(text):
                    items[item] = text

    items.index.name = 'item'

    return items


def identify_sections(inpath=filing_path, outpath=sections_path):
    """
    Loop over filing_path parsed filings and perform section identification
    Write results in sections_path
    """
    if not outpath.exists():
        outpath.mkdir(exist_ok=True, parents=True)

    for i, filing in enumerate(inpath.glob('*.txt'), 1):
        filing_id = int(filing.stem)
        sections_file = outpath / (filing.stem + '.csv')
        if sections_file.exists():
            continue

        if i % 500 == 0:
            print(i, end=' ', flush=True)

        txt = filing_section_identification(doc=filing.read_text())

        # persist data
        txt.to_csv(sections_file)


def preprocessor(doc):
    """
    Preprocesing pipeline:
        remove stopwrods
        remove numbers or not alphanumeric
        remove punctuation
        remove spaces
        remove pronoums
        remove punctuation, symbols and other  https://universaldependencies.org/docs/u/pos/
    lower case strings and filter out short sentences
    :param doc: SpaCy document
    :return: clean sentences and lemmatized documents as strings
    """
    clean_doc = []
    cleand_doc_lemma = []
    for t, token in enumerate(doc, 1):
        token_lemma = token.lemma_
        if not any([token.is_stop,
                    token.is_digit,
                    not token.is_alpha,
                    token.is_punct,
                    token.is_space,
                    token_lemma == '-PRON-',
                    token.pos_ in ['PUNCT', 'SYM', 'X']]):

            clean_doc.append(token.text.lower())
            cleand_doc_lemma.append(token_lemma)

    return " ".join(clean_doc), " ".join(cleand_doc_lemma)


def doc_sentence_preprocessor(doc, min_sentece_length: int = 0):
    """
    Apply sentence split and preprocessor to each document
    :param doc: SpaCy document resulting from appliying a pipeline
    :param min_sentece_length: min sentence length (after cleaning)
    :return: clean sentences as pandas Series, lemmatized documents as string
    """
    clean_sentences = {}
    clean_lemma_sentences = []
    for s, sentence in enumerate(doc.sents):
        if sentence is not None:
            clean_sentence, clean_sentence_lemma = preprocessor(sentence)

            if len(clean_sentence) > min_sentece_length:
                clean_sentences[s] = clean_sentence
                clean_lemma_sentences.append(clean_sentence_lemma)

    clean_sentences = pd.Series(clean_sentences, name='text')
    clean_sentences.index.name = 'sentence'

    return clean_sentences, clean_lemma_sentences


def parse_sections(nlp, sections: list, text_col= 'text', item_col='item',
                   inpath=sections_path, outpath_sentences=clean_path, outpath_docs=lemma_path,):
    """
    For each filing (splited by section), create a sentences file and a lemmatized document
    :param nlp: SpaCy language model
    :param sections: a list of filing sections of interest
    """
    if not outpath_sentences.exists():
        outpath_sentences.mkdir(exist_ok=True)

    if not outpath_docs.exists():
        outpath_docs.mkdir(exist_ok=True)
    # For each section_doc, create a pandas DF where each row is a sentence
    # item, sentence, text
    # Also, dump lemmatized section texts to corpus of files

    start = time()
    to_do = len(list(inpath.glob('*.csv')))
    done = len(list(clean_path.glob('*.csv'))) + 1
    # loop over filings
    for text_file in inpath.glob('*.csv'):
        file_id = int(text_file.stem)
        clean_file = outpath_sentences / f'{file_id}.csv'
        lemma_file = outpath_docs / f'{file_id}.txt'
        if clean_file.exists() and lemma_file.exists():
            continue
        # Read and filter fillings (each row is a section)
        items = pd.read_csv(text_file).dropna()
        items[item_col] = items.item.astype(str)
        items = items[items[item_col].isin(sections)]
        if done % 100 == 0:
            duration = time() - start
            to_go = (to_do - done) * duration / done
            print(f'{done:>5}\t{format_time(duration)}\t{format_time(to_go)}')

        # Clean sections and extract sentences and lemmatized document
        clean_sections = dict()
        lemma_doc = []
        # for each filling, loop over items (sections)
        for idx, row in items.iterrows():
            item = row[item_col]
            text = row[text_col]
            doc = nlp(text)
            clean_sentence, clean_lemma_sentence = doc_sentence_preprocessor(doc, min_sentece_length=0)
            clean_sections[item] = clean_sentence
            lemma_doc.append(" ".join(clean_lemma_sentence))

        # persist data
        # Section sentences
        if len(clean_sections) > 0:
            persist_df = pd.concat(clean_sections)
            persist_df.index.names = ['item', 'sentence']
            persist_df.dropna().to_csv(clean_file)
        # Join lemmatized sections into a document
        if len(lemma_doc) > 0:
            lemma_file.write_text(" ".join(lemma_doc), encoding='utf-8')

        done += 1