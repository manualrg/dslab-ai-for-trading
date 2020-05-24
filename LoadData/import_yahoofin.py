from datetime import datetime
import lxml
from lxml import html
import requests
import numpy as np
import pandas as pd

def get_page(url):
    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'
    }
    page = requests.get(url, headers)
    return page


def parse_rows(table_rows):
    parsed_rows = []

    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")

        none_count = 0

        for rs in el:
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                parsed_row.append(np.NaN)
                none_count += 1

        if (none_count < 4):
            parsed_rows.append(parsed_row)

    return pd.DataFrame(parsed_rows)


def clean_data(df):
    df = df.set_index(0)  # Set the index to the first column: 'Period Ending'.
    df = df.transpose()  # Transpose the DataFrame, so that our header contains the account names

    # Rename the "Breakdown" column to "Date"
    df.rename(inplace=True, columns={"Breakdown": "Date"})
    # cols = list(df.columns)
    # cols[0] = 'Date'
    # df = df.set_axis(cols, axis='columns', inplace=False)
    df['Date'] = pd.to_datetime(df['Date'])

    numeric_columns = list(df.columns)[1::]  # Take all columns, except the first (which is the 'Date' column)

    for column_index in range(1, len(df.columns)):  # Take all columns, except the first (which is the 'Date' column)
        df.iloc[:, column_index] = df.iloc[:, column_index].str.replace(',', '')  # Remove the thousands separator
        df.iloc[:, column_index] = df.iloc[:, column_index].astype(np.float64)  # Convert the column to float64

    return df


def scrape_table(url):
    # Fetch the page that we're going to parse
    page = get_page(url);

    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)
    #print(tree.xpath("//h1/text()"))
    # Fetch all div elements which have class 'D(tbr)'
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")

    # Ensure that some table rows are found; if none are found, then it's possible
    # that Yahoo Finance has changed their page layout, or have detected
    # that you're scraping the page.
    assert len(table_rows) > 0

    df = parse_rows(table_rows)
    df = clean_data(df)

    return df

def iter_tickers(tickers, table_name, dump_path):
    assert table_name in ['balance-sheet',
                          'financials',  #income statements
                          'cash-flow'
                         ]
    import os, time
    assert os.path.isdir(dump_path), print('dump_path' + dump_path + "does not exist")
    np.random.seed(123)
    sleep_secs = 10
    df_lst = []
    dump_registry_cols = ['table-name', 'n_rows', 'n_cols', 'exe_dt']

    for i, ticker in enumerate(tickers):
        df = scrape_table('https://finance.yahoo.com/quote/' + ticker + '/' + table_name +'?p=' + ticker)
        df.insert(0, 'ticker', ticker)
        df.insert(1, 'table-name', table_name)
        df_lst.append(df)

        n_rows, n_cols = df.shape
        row = [table_name, n_rows, n_cols, pd.Timestamp.now()]
        if i == 0:
            dump_registry_df = pd.DataFrame(index=[ticker], data=dict(zip(dump_registry_cols, row)))
        else:
            dump_registry_df.loc[ticker, dump_registry_cols] = row

        dump_name = table_name + ticker + '.csv'
        df.to_csv(dump_path + dump_name, index=False)

        time.sleep(np.random.choice(range(1, sleep_secs+1)))

    dump_registry_df.to_csv(dump_path + 'yahoofin' + table_name + '.csv', index_label='ticker')
    data = pd.concat(df_lst, axis=0)#, sort=False)
    data.set_index(inplace=True, keys=['Date', 'ticker'])
    return data