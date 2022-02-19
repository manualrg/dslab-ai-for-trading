import matplotlib.pyplot as plt
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from src.load_data import io_utils

class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    def get(self, url):
        """
        Performs actual donwloading of a SEC filling, given a index-url
        :param url:
        :return: filling text
        """
        return self._call_sec(url).text

def get_sec_data(sec_api, cik, doc_type, start=0, count=60):
    """
    Download SEC index urls for a set of tickers (cik) in safe way (api rate limit)
    :param sec_api:  SecAPI instance
    :param cik: dictionary of tickers-cik (SEC Central Key Index)
    :param doc_type: SEC document type (like 10-Ks)
    :param start:
    :param count:
    :return: dict(ticker: (index_url, file_type, file_date))
    """

    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed

    if feed is None:
        print(f'Missing CIK: {cik} filling: {doc_type}')
    else:
        entries = []
        for entry in feed.find_all('entry', recursive=False):
            entry_tup = (
                entry.content.find('filing-href').getText(),
                entry.content.find('filing-type').getText(),
                entry.content.find('filing-date').getText()
            )
            entries.append(entry_tup)
        return entries


def get_documents(text):
    """
    Each filling is broken into several associated documents, sectioned off in the fillings with the tags:
      <DOCUMENT> </DOCUMENT>
    Extract the documents from the text

    Parameters
    ----------
    text : str
        The text with the document strings inside

    Returns
    -------
    extracted_docs : list of str
        The document strings found in `text`
    """

    # Get start and end chunks
    regex_start = re.compile(r"<DOCUMENT>")
    regex_end = re.compile(r"</DOCUMENT>")

    start_chunks = regex_start.finditer(text)
    end_chunks = regex_end.finditer(text)

    # Extract documents from text
    raw_extracted_docs = []
    for start_match, end_match in zip(start_chunks, end_chunks):
        start_idx = start_match.end()
        end_idx = end_match.start()
        extracted_doc = text[start_idx:end_idx]
        raw_extracted_docs.append(extracted_doc)

    return raw_extracted_docs


def get_document_type(doc):
    """
    The document type is located on a line with the <TYPE> tag.
    Return the document type upcased

    Parameters
    ----------
    doc : str
        The document string

    Returns
    -------
    doc_type : str
        The document type upcased
    """

    regex_type = re.compile(r'(<TYPE>)(.*)')
    matches = regex_type.finditer(doc)
    doc_type = [match.group(2) for match in matches]

    return doc_type[0].upper()

from zipfile import ZipFile
import shutil
import gzip

def run_download_and_parse(sec_data, sec_api, path, doc_type, oldest_filling_date: str = '2000-01-01'):
    """

    :param sec_data:
    :param sec_api:
    :param path:
    :param doc_type:
    :param oldest_filling_date:
    :return:
    """
    #staging_path = os.path.join(path, 'staging', '')
    #if not os.path.isdir(staging_path):
    #    os.mkdir(staging_path)

    start_dt = pd.Timestamp(oldest_filling_date)
    doc_type_path = doc_type.replace('-', '').lower()
    extension = 'gz'  # no extension point

    for ticker, data in sec_data.items():
        ticker_path = ticker.lower()
        for index_url, file_type, file_date in tqdm(data, desc=f'Downloading {ticker} Fillings', unit='filling'):
            file_dt = pd.Timestamp(file_date)
            file_date_path = file_date.replace('-', '')
            if (file_type == doc_type):
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')

                raw_filling = sec_api.get(file_url)
                raw_extracted_docs = get_documents(raw_filling)
                for document in raw_extracted_docs:
                    outfilename = f"{path}{ticker_path}_{doc_type_path}_{file_date_path}.{extension}"
                    if not os.path.isfile(outfilename):
                        if (get_document_type(document) == doc_type) and (file_dt >= start_dt):
                            with gzip.GzipFile(outfilename, "wb") as gzip_text_file:
                                gzip_text_file.write(document.encode())


def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')


def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label)
        plt.legend()
        plt.xticks(rotation=90)

    plt.show()

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

def get_cik_mapping():
    """
    SEC CIK to ticker mapping
    https://www.sec.gov/include/ticker.txt
    :return: pandas DataFrame
    """
    path_sec_cik = os.path.join(io_utils.raw_path, 'sec_fillings', '')
    sec_cik_df = pd.read_csv(path_sec_cik + 'sec_cik.csv')
    sec_cik_df['cik'] = sec_cik_df['cik'].apply(lambda x: f'{x:010d}')

    return sec_cik_df


cik_lookup = {
    'AMZN': '0001018724',
    'BMY': '0000014272'}

a = {'CNP': '0001130310',
    'CVX': '0000093410',
    'FL': '0000850209',
    'FRT': '0000034903',
    'HON': '0000773840'}

additional_cik = {
    'AEP': '0000004904',
    'AXP': '0000004962',
    'BA': '0000012927',
    'BK': '0001390777',
    'CAT': '0000018230',
    'DE': '0000315189',
    'DIS': '0001001039',
    'DTE': '0000936340',
    'ED': '0001047862',
    'EMR': '0000032604',
    'ETN': '0001551182',
    'GE': '0000040545',
    'IBM': '0000051143',
    'IP': '0000051434',
    'JNJ': '0000200406',
    'KO': '0000021344',
    'LLY': '0000059478',
    'MCD': '0000063908',
    'MO': '0000764180',
    'MRK': '0000310158',
    'MRO': '0000101778',
    'PCG': '0001004980',
    'PEP': '0000077476',
    'PFE': '0000078003',
    'PG': '0000080424',
    'PNR': '0000077360',
    'SYY': '0000096021',
    'TXN': '0000097476',
    'UTX': '0000101829',
    'WFC': '0000072971',
    'WMT': '0000104169',
    'WY': '0000106535',
    'XOM': '0000034088'}