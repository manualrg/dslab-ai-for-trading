import matplotlib.pyplot as plt
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

from ratelimit import limits, sleep_and_retry

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

def get_sec_data(sec_api, cik, newest_pricing_data, doc_type, start=0, count=60):
    """
    Download SEC index urls for a set of tickers (cik) in safe way (api rate limit)
    :param sec_api:  SecAPI instance
    :param cik: dictionary of tickers-cik (SEC Central Key Index)
    :param newest_pricing_data: Latest date to request data
    :param doc_type: SEC document type (like 10-Ks)
    :param start:
    :param count:
    :return: dict(ticker: (index_url, file_type, file_date))
    """
    newest_pricing_data = pd.to_datetime(newest_pricing_data)
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)
        if pd.to_datetime(entry.content.find('filing-date').getText()) <= newest_pricing_data]

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

# {ticker: cik}
cik_lookup = {
    'AMZN': '0001018724',
    'BMY': '0000014272',
    'CNP': '0001130310',
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