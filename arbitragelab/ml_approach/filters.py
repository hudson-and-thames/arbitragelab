import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def get_rolling_correlation(frame, lookback, scale=False):
    two_legged_df = frame.iloc[:, 0:2] 
    
    daily_corr = two_legged_df.rolling(lookback, min_periods=lookback).corr()
    daily_corr = daily_corr.iloc[:, 0].reset_index().dropna()
    
    final_corr = daily_corr[daily_corr['level_1'] == two_legged_df.columns[1]]
    final_corr.set_index('date', inplace=True)
    final_corr.drop(['level_1'], axis=1, inplace=True)
    final_corr.dropna(inplace=True)
    
    if scale:
        scaler = MinMaxScaler()
        scaled_corr = scaler.fit_transform(final_corr.iloc[:, 0].values.reshape(-1, 1))#.diff()
        corr_series = pd.Series(data=scaled_corr.reshape(1, -1)[0], index=final_corr.index) 
        return corr_series
    
    return final_corr
    

def get_events_by_corr_filter(frame: pd.DataFrame,
                              lookback=30,
                              buy_threshold=0.4,
                              sell_threshold=0.8):
    
    two_legged_df = frame.iloc[:, 0:2] 
    corr_series = get_rolling_correlation(two_legged_df, lookback, scale=True).diff()
    
    buy_signal = frame.index.isin( corr_series[corr_series > buy_threshold].index )
    sell_signal = frame.index.isin( corr_series[corr_series < sell_threshold].index )
    
    frame['side'] = 0
    frame.loc[buy_signal, 'side'] = 1 
    frame.loc[sell_signal, 'side'] = -1 
    frame['side'] = frame['side'].shift(1)
    
    return frame



def get_events_by_asym_filter(spread: pd.Series, 
                              buy_threshold: int, 
                              sell_threshold: int,
                              p_1: float, p_2: float) -> pd.DataFrame:
    
    buy_threshold = abs(p_1)*buy_threshold
    sell_threshold = abs(p_2)*sell_threshold
  
    return get_events_by_threshold_filter(spread=spread, 
                                          buy_threshold=buy_threshold, 
                                          sell_threshold=sell_threshold)


def get_events_by_threshold_filter(spread: pd.Series, 
                             buy_threshold: int, 
                             sell_threshold: int) -> pd.DataFrame:
    
    frame = spread.copy().to_frame()
    
    buy_signal = frame.index.isin( spread[spread < buy_threshold].index )
    sell_signal = frame.index.isin( spread[spread > sell_threshold].index )
    
    frame['side'] = 0
    frame.loc[buy_signal, 'side'] = 1 
    frame.loc[sell_signal, 'side'] = -1 
    frame['side'] = frame['side'].shift(1)
    
    return frame


def plot_events(frame: pd.DataFrame):
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(211)
    plt.plot(frame.iloc[:, 0].cumsum())
    for trade_evnt in frame[frame['side'] == 1].index:
        plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
    
    plt.subplot(212)
    plt.plot(frame.iloc[:, 0].cumsum())
    for trade_evnt in frame[frame['side'] == -1].index:
        plt.axvline(trade_evnt, color="tab:red", alpha=0.2)

def plot_corr(two_legged_df: pd.Series, lookback: int = 30, 
                  buy_threshold: int = 0.2, sell_threshold: int = 0.9):

    spread = two_legged_df.iloc[:, 0] - two_legged_df.iloc[:, 1]
    
    corr_series = get_rolling_correlation(two_legged_df, lookback, scale=False)
    
    corr_events = get_events_by_corr_filter(two_legged_df, lookback=lookback, 
                                            buy_threshold=buy_threshold, 
                                            sell_threshold=sell_threshold)['side']
    
    plt.figure(figsize=(15,10))
    plt.subplot(311)
    plt.plot(corr_series.diff())
    plt.axhline(y=buy_threshold, color='g', linestyle='--')
    plt.axhline(y=sell_threshold, color='r', linestyle='--')
    
    plt.subplot(312)
    plt.plot(spread)
    for trade_evnt in spread[corr_events == 1].index:
        plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
        
    plt.subplot(313)
    plt.plot(spread)
    for trade_evnt in spread[corr_events == -1].index:
        plt.axvline(trade_evnt, color="tab:red", alpha=0.2)


def get_events_by_cusum_corr_filter(frame: pd.DataFrame,
                              lookback=30,
                              buy_threshold=0.1,
                              sell_threshold=0.9):
    
    two_legged_df = frame.iloc[:, 0:2] 
    
    corr_series = get_rolling_correlation(two_legged_df, lookback, scale=True)
    
    buy_event_dates = ml.filters.cusum_filter(corr_series, threshold=buy_threshold)
    buy_signal = frame.index.isin(buy_event_dates)
    
    sell_event_dates = ml.filters.cusum_filter(abs(corr_series-1), threshold=sell_threshold)
    sell_signal = frame.index.isin(sell_event_dates)

    frame['side'] = 0
    frame.loc[buy_signal, 'side'] = 1 
    frame.loc[sell_signal, 'side'] = -1 
    frame['side'] = frame['side'].shift(1)
    
    return frame

def plot_corr_cusum(two_legged_df: pd.Series,lookback: int = 30, 
                    buy_threshold=0.1, sell_threshold=0.9):
    
    spread = two_legged_df.iloc[:, 0] - two_legged_df.iloc[:, 1]

    corr_series = get_rolling_correlation(two_legged_df, lookback, scale=True)
    corr_events = get_events_by_cusum_corr_filter(two_legged_df, lookback=lookback, 
                                                  buy_threshold=buy_threshold, 
                                                  sell_threshold=sell_threshold)['side']

    plt.figure(figsize=(15,10))
    plt.subplot(411)
    plt.plot(corr_series)
    for trade_evnt in spread[corr_events == 1].index:
        plt.axvline(trade_evnt, color="tab:green", alpha=0.2)
    
    plt.subplot(412)
    plt.plot(spread)
    for trade_evnt in spread[corr_events == 1].index:
        plt.axvline(trade_evnt, color="tab:green", alpha=0.2)

    plt.subplot(413)
    plt.plot(corr_series)
    for trade_evnt in spread[corr_events == -1].index:
        plt.axvline(trade_evnt, color="tab:red", alpha=0.2)
        
    plt.subplot(414)
    plt.plot(spread)
    for trade_evnt in spread[corr_events == -1].index:
        plt.axvline(trade_evnt, color="tab:red", alpha=0.2)
