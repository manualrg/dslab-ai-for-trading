import re
import pandas as pd

def get_filling_doc(filling: str, target_doc: str = '10-K'):
    """
    HTML data from 10-K document from SEC website
    doc_start_is: index list containing start 10-K start tag :<DOCUMENT>
    doc_end_is: index list containing start 10-K end tag:  </DOCUMENT> t
    doc_types: Each section within the document tags is clearly marked by a <TYPE> tag followed by the name of the section
    :param filling:  10-K HTML doc
    :param target_doc: target doc label
    :return: filling section specified by target_doc
    """
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(filling)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(filling)]
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(filling)]

    document = {}
    print(doc_start_is, doc_end_is, doc_types)
    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start_i, doc_end_i in zip(doc_types, doc_start_is, doc_end_is):
        print(doc_type, doc_start_i, doc_end_i)
        if doc_type == target_doc:
            document[doc_type] = filling[doc_start_i:doc_end_i]

    return document


def get_10k_risk_sections_df(text: str, file: str):
    """
    Match all four patterns for Items 1A, 7, and 7A. Item 1A can be found in either of the following patterns:
        >Item 1A
        >Item&#160;1A
        >Item&nbsp;1A
        >ITEM 1A
    Pandas dataframe .drop_duplicates() method to only keep the last Item matches in the dataframe and drop the rest.
    Remove the Item matches that correspond to the index. In the code below use the Pandas dataframe
    .drop_duplicates() method to only keep the last Item matches in the dataframe and drop the rest.
    :param doc: documents
    :param target_doc: doc type to filter
    :return: Pandas dataframe with the following column names: 'start','end', 'next_start' and 'item' as index
    """

    re_risk = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|7A|7)\.{0,1})|(ITEM\s(1A|7A|7))')
    matches = re_risk.finditer(text)
    pos_dat_columns = ['item', 'start', 'end']
    try:
        first_match = next(matches)
    except:
        print(f"No match in file: {file}")  # action for no match
        test_df = pd.DataFrame(data=[['all', 0, 0]], columns=pos_dat_columns)
    else:
        test_df = pd.DataFrame(data=[(x.group(), x.start(), x.end()) for x in matches], columns=pos_dat_columns)
        test_df['item'] = test_df.item.str.lower()

        # Get rid of unnecessary characters from the dataframe
        test_df.replace('&#160;', ' ', regex=True, inplace=True)
        test_df.replace('&nbsp;', ' ', regex=True, inplace=True)
        test_df.replace(' ', '', regex=True, inplace=True)
        test_df.replace('\.', '', regex=True, inplace=True)
        test_df.replace('>', '', regex=True, inplace=True)

    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')
    pos_dat.set_index('item', inplace=True)
    pos_dat['next_start'] = pos_dat['end'].shift(-1)
    try:
        pos_dat.iloc[-1, -1] = len(text)
    except:
        print(f"Empty pos_dat in  file: {file}")
        pos_dat = pd.DataFrame(index=['all'], data=[[0, 0, len(text)]], columns=['start', 'end', 'next_start'])

    return pos_dat

def get_section_text(text: str, pos_dat: pd.DataFrame):
    """

    :param documents: {'doc_type':  document}
    :param pos_dat: Pandas dataframe with the following column names: 'start','end', 'next_start' and 'item' as index
    :param target_doc: doc type to filter
    :return: dictionary of {'section': document}
    """

    assert set(['start', 'end', 'next_start']).issubset(pos_dat.columns), "pos_dat does not have specified columns"

    docs = {}
    for idx_section, row in pos_dat.iterrows():
        docs[idx_section] = text[int(row['end']):int(row['next_start'])]

    return docs

import os
import gzip
from bs4 import BeautifulSoup

def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()

    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)

    return text

import datetime as dt

def get_risk_sections_and_parse(inpath, outpath, write_gzip=True):
    """

    :param inpath:
    :param outpath:
    :param write_gzip:
    :return:
    """
    in_listdir = os.listdir(inpath)
    control_lst = []
    n_exist = 0
    for file in in_listdir:
        ticker, doc_type, date = file.split("_")
        date = date.split(".")[0]
        infilename = inpath + file

        if write_gzip:
            outfilename = outpath + file
        else:
            outfilename = outpath + file.split(".")[0] + ".txt"

        if os.path.isfile(outfilename):
            n_exist += 1
        else:
            with gzip.open(infilename, "rb") as f:
                doc = f.read()
            doc = doc.decode()

            tenk_risk_pos_dat = get_10k_risk_sections_df(text=doc, file=file)
            tenk_risk_sections = get_section_text(text=doc, pos_dat=tenk_risk_pos_dat)

            sections = []
            for item, section in tenk_risk_sections.items():
                section_clean = clean_text(section)
                sections.append(section_clean)
            doc_clean = " ".join(sections)

            if write_gzip:
                with gzip.GzipFile(outfilename, "wb") as gzip_text_file:
                    gzip_text_file.write(doc_clean.encode())
            else:
                with open(outfilename, "w") as text_file:
                    text_file.write(doc_clean)

            tenk_risk_pos_dat['ticker'] = ticker
            tenk_risk_pos_dat['doc_type'] = doc_type
            tenk_risk_pos_dat['date'] = dt.datetime.strptime(date, "%Y%m%d")

            control_lst.append(tenk_risk_pos_dat)
    print(f'Number of files that existed previously: {n_exist}')

    return pd.concat(control_lst, axis=0)