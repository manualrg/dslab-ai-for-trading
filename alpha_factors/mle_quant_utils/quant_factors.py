import numpy as np
import pandas as pd
import alphalens as al
import datetime as dt

from zipline.pipeline.factors import CustomFactor, DailyReturns, Returns, SimpleMovingAverage
from zipline.pipeline.data import USEquityPricing

# Region alpha-eval
def compute_sharpe_ratio(df, frequency="daily"):
    if frequency == "daily":
        annualization_factor = np.sqrt(252)
    elif frequency == "monthly":
        annualization_factor = np.sqrt(12)
    else:
        annualization_factor = 1

    return annualization_factor * df.mean() / df.std()

def factor_evaluation(factor_data, factor_names, frequency="daily"):
    ls_sharpe, ls_factor_return, ls_rank_ic, ls_fra, ls_qr = [], [], [], [], []
    unixt_factor_data = {}

    for i, factor_name in enumerate(factor_names):
        print("Calculating the factor weighted returns and sharpe-ratio for: " + factor_name)
        factor_return = al.performance.factor_returns(factor_data[factor_name])
        factor_return.columns = [factor_name]
        ls_factor_return.append(factor_return)

        sharpe_ratio = compute_sharpe_ratio(ls_factor_return[i], frequency).to_frame('Sharpe ratio')
        ls_sharpe.append(sharpe_ratio)

        print("Calculating the Ranked IC for: " + factor_name)
        rank_ic = al.performance.factor_information_coefficient(factor_data[factor_name])
        rank_ic.columns = [factor_name]
        ls_rank_ic.append(rank_ic)

        unixt_index_data = [(x.timestamp(), y) for x, y in factor_data[factor_name].index.values]
        unixt_factor_data[factor_name] = factor_data[factor_name].set_index(
            pd.MultiIndex.from_tuples(unixt_index_data, names=['date', 'asset']))

        print("Calculating the FRA for: " + factor_name)
        fra = al.performance.factor_rank_autocorrelation(unixt_factor_data[factor_name]).to_frame()
        fra.columns = [factor_name]
        ls_fra.append(fra)

        print("Calculating Quantile returns for: " + factor_name)
        quantile_return, quantile_stderr = al.performance.mean_return_by_quantile(unixt_factor_data[factor_name])
        quantile_return.columns = [factor_name]
        ls_qr.append(quantile_return)

        df_factor_return = pd.concat(ls_factor_return, axis=1)
        df_sharpe = pd.concat(ls_sharpe, axis=0)
        df_rank_ic = pd.concat(ls_rank_ic, axis=1)
        df_fra = pd.concat(ls_fra, axis=1)
        df_qr = pd.concat(ls_qr, axis=1)

        df_fra.index = [dt.datetime.fromtimestamp(x) for x in df_fra.index]
    return df_factor_return, df_sharpe, df_rank_ic, df_fra, df_qr

# Region alpha_factors
def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe).demean(groupby=sector).rank().zscore()


def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    return -Returns(window_length=window_length, mask=universe).demean(groupby=sector).rank().zscore()


def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    unsmoothed_factor = mean_reversion_5day_sector_neutral(window_length, universe, sector)
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=window_length).rank().zscore()


class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """
    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    return TrailingOvernightReturns(inputs=[cto_out],
                                    window_length=trail_overnight_returns_window_length).rank().zscore()


def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    unsmoothed_factor = overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe)
    return SimpleMovingAverage(inputs=[unsmoothed_factor],
                               window_length=trail_overnight_returns_window_length).rank().zscore()

# Region quant-features

class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        mean_returns = np.nanmean(returns)
        out[:] = np.sqrt(np.nanmean((returns - mean_returns) ** 2))


class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1  # We'll want to set this in the constructor when creating the object.
    window_safe = True

    def compute(self, today, assets, out, returns):
        DAILY_TO_ANNUAL_SCALAR = 252.  # 252 trading days in a year
        """
        For each row (each row represents one day of returns), 
        calculate the average of the cross-section of stock returns
        So that market_returns has one value for each day in the window_length
        So choose the appropriate axis (please see hints above)
        """
        mkt_returns = np.nanmean(returns, axis=1)

        """ 
        Calculate the mean of market returns
        """
        mkt_returns_mu = np.nanmean(mkt_returns)

        """
        Calculate the standard deviation of the market returns, then annualize them.
        """
        out[:] = np.sqrt(DAILY_TO_ANNUAL_SCALAR * np.nanmean((mkt_returns - mkt_returns_mu) ** 2))
