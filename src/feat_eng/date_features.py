import pandas as pd

def feat_is_month(date_s, month_id):
    assert isinstance(date_s, pd.DatetimeIndex)

    return (date_s.month == month_id).astype(int)

def feat_is_target_date(date_s, date_subset):
    assert isinstance(date_s, pd.DatetimeIndex)

    return (date_s.isin(date_subset)).astype(int)

def compute_date_features(factor_df, start_dt, end_dt):

    assert isinstance(factor_df.index, pd.MultiIndex), "factor_df should be a MultiIndex"
    date_s = factor_df.index.get_level_values(0)
    assert isinstance(date_s, pd.DatetimeIndex), \
        "factor_df level(0) index should be DateTimeIndex"

    date_features = pd.DataFrame(index=factor_df.index, dtype=int)

    date_features['is_January'] = feat_is_month(date_s, 1)
    date_features['is_December'] = feat_is_month(date_s, 12)
    date_features['month_end'] = date_s.isin(pd.date_range(start=start_dt, end=end_dt, freq='BM')).astype(int)
    date_features['month_start'] = date_s.isin(pd.date_range(start=start_dt, end=end_dt, freq='BMS')).astype(int)
    date_features['qtr_end'] = date_s.isin(pd.date_range(start=start_dt, end=end_dt, freq='BQ')).astype(int)
    date_features['qtr_start'] = date_s.isin(pd.date_range(start=start_dt, end=end_dt, freq='BQS')).astype(int)


    weekday_s = pd.Series(index=factor_df.index, data=date_s.weekday)
    weekday_ohe_df = pd.get_dummies(weekday_s, prefix='weekday')
    weekday_ohe_cols = weekday_ohe_df.columns.tolist()
    date_features[weekday_ohe_cols] = weekday_ohe_df

    quarter_s = pd.Series(index=factor_df.index, data=date_s.quarter)
    qtr_ohe_df = pd.get_dummies(quarter_s, prefix='qtr')
    qtr_ohe_cols = qtr_ohe_df.columns.tolist()
    date_features[qtr_ohe_cols] = qtr_ohe_df

    return date_features
