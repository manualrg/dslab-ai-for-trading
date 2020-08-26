import logging

import quandl
from pandas_datareader import data as pdr
import simfin as sf

log = logging.getLogger(__name__)



def load_tbills(path: str, filename: str, start_dt: str, end_dt: str):
    """
    https://docs.quandl.com/docs/python-time-series
    :param start_dt: Start date
    :param end_dt: End Date
    """
    quant_tk = os.environ.get("QUANDL")

    log.info(msg="Downloading data from Quandl: USTREASURY/YIELD")
    us_tbills = quandl.get("USTREASURY/YIELD", authtoken=quant_tk, start_date=start_dt, end_date=end_dt)
    us_tbills['3 MO rets'] = us_tbills['3 MO'] * 365 / 90 / 100
    us_tbills['1 YR rets'] = us_tbills['1 YR'] / 100

    us_tbills.to_csv(path + filename)

    return us_tbills

def load_sp500(path: str, filename: str, start_dt: str, end_dt: str):

    sp500 = pdr.DataReader(['sp500'], 'fred', start_dt, end_dt)
    sp500.rename(inplace=True, columns={'sp500': 'sp500_close'})
    sp500['sp500_ret'] = sp500['sp500_close'].pct_change()

    sp500.to_csv(path + filename)

    return sp500

def load_financial_data(path: str):

    sf.set_api_key('free')

    sf.set_data_dir(path)

    # Load the full list of companies in the selected market (United States).
    df_companies = sf.load_companies(market='us')

    # Load all the industries that are available.
    df_industries = sf.load_industries()

    # Load the quarterly Income Statements for all companies in the selected market.
    df_income = sf.load_income(variant='quarterly', market='us')

    # Load the quarterly Balance Sheet data for all companies in the selected market.
    df_balance = sf.load_balance(variant='quarterly', market='us')

    # Load the quarterly Balance Sheet data for all companies in the selected market.
    df_cashflow = sf.load_cashflow(variant='quarterly', market='us')

    return df_companies, df_industries, df_income, df_balance, df_cashflow


if __name__ == '__main__':
    from src.load_data import io_utils
    import os

    mkt_data_path = os.path.join(io_utils.raw_path, "market_data", "")
    simfin_path = os.path.join(io_utils.raw_path, "sinfim_data", "")

    us_tbills = load_tbills(mkt_data_path, "quandl_ustbills.csv", "2010-01-01", "2020-05-05")
    sp500 = load_sp500(mkt_data_path, "fred_sp500.csv", "2010-01-01", "2020-05-05")
    df_companies, df_industries, df_income, df_balance, df_cashflow = load_financial_data(simfin_path)