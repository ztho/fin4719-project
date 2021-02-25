import numpy as np 
import pandas as pd 

import data_loader as data 
from utils import *

def show_vr_stats(data, price, k_list, var_form):
    vr_df = test_multiple_periods(data, price, k_list, var_form).drop(columns = ["stat"])
    
    
    #highlight significant 
    # def highlight_significant(s):
    #     df = s.copy()
    #     mask = df['pval'] < .1
    #     df.loc[mask,:] = 'background-color: yellow'
    
    # vr_df = vr_df.style.applymap(highlight_significant)

    vr_df = vr_df.rename(columns = {"k": "K-Period", 
                                    "pval": "P-Value",
                                    "vr": "Variance Ratio"})
    
    vr_df = vr_df[["K-Period", "Variance Ratio", "P-Value"]]
    return vr_df

def show_recent_returns(df_ticker, ticker, num_days):
    ret = get_recent_returns(df_ticker, ticker, num_days)
    ret_form = str(round(ret * 100, 1)) + "%"
    return ret_form

def show_recent_vols(df_ticker, ticker, num_days):
    vols = get_recent_vol(df_ticker, ticker, num_days)
    vols_form = str(round(vols *100, 2)) + "%"
    return vols_form 

def show_if_random_walk(df_ticker, ticker, k_list, var_form, p_val):
    vr_stats = test_multiple_periods(df_ticker, ticker, k_list, var_form)
    is_random_walk = False if all(p > p_val for p in vr_stats['pval'].values) else True
    return "Yes" if is_random_walk else "No"
