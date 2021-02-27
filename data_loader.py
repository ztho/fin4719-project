# Module to load data 
import pandas as pd
import numpy as np 

# Prepare etf data for portfolio optimization
def get_etf_data():
    """
    Loads the ETF Dataset

    :returns: pd.DataFrame
    """
    etf_data = pd.read_csv("data_files/etf_data.csv")
    etf_data = etf_data.set_index("Date")
    etf_data = etf_data.iloc[::-1]
    etf_data = etf_data.replace("-", np.nan)
    etf_data = etf_data.dropna()
    etf_data = etf_data.astype(np.float64)
    return etf_data


# Prepare selected stock data for TAA allocation
def get_stock_data():
    """
    Loads the individual stocks dataset

    :returns: pd.DataFrame
    """
    sel_stock_data = pd.read_csv("data_files/selected_stock_data_3.csv").set_index("Date")
    sel_stock_data = sel_stock_data.astype(np.float64)
    sel_stock_data.index = pd.to_datetime(sel_stock_data.index) # set index to datetime format
    return sel_stock_data

def get_rf_rate():
    """
    Loads historical US 10-Year Treasury Rates

    :returns: pd.DataFrame
    """
    us_rf = pd.read_csv("data_files/us_daily_rf_rate.csv")
    us_rf = us_rf.set_index("Date")
    us_rf.index = pd.to_datetime(us_rf.index) # set index to datetime format
    us_rf = us_rf.rename(columns = {"Rate" : "rf"})
    us_rf['rf'] = us_rf['rf'].astype(int) / 100 
    return us_rf

# get S&P 500 daily data (retrieved above)
def get_snp():
    """
    Loads the historical prices of the S&P 500 index
    
    :returns: pd.DataFrame
    """
    gspc = pd.read_csv("data_files/^GSPC.csv")
    gspc = gspc.rename(columns = {"Adj Close" : "^GSPC"})
    gspc = gspc.set_index("Date")
    gspc.index = pd.to_datetime(gspc.index)
    return gspc

# Merge data with dataframe 
def get_stock_with_benchmarks():
    sel_stock_data = get_stock_data()
    gspc = get_snp()
    us_rf = get_rf_rate() 
    
    sel_stock_data_with_benchmarks = sel_stock_data.dropna().merge(gspc, left_index = True, right_index = True)
    sel_stock_data_with_benchmarks = sel_stock_data_with_benchmarks.merge(us_rf, left_index = True, right_index = True).astype(np.float64)
    return sel_stock_data_with_benchmarks

def get_rel_perf():
    return pd.read_csv("data_files/rel_perf.csv").set_index("ticker")