import streamlit as st

import numpy as np 
import pandas as pd 

import data_loader as data 
from utils import *
import user_state as u

# from bokeh.plotting import figure, output_file, show
# from bokeh.models import ColumnDataSource
# from bokeh.models.tools import HoverTool
# from bokeh.layouts import gridplot
# from bokeh.palettes import Spectral3


def show_vr_stats(data, price, k_list, var_form, pval_thres = .05):
    vr_df = test_multiple_periods(data, price, k_list, var_form).drop(columns = ["stat"])
    
    def highlight(x, pval_thres):
        if x['P-Value'] < pval_thres:
            return ['background-color: palegreen'] * 3
        else:
            return ['background-color: white'] * 3

    vr_df = vr_df.rename(columns = {"k": "K-Period", 
                                    "pval": "P-Value",
                                    "vr": "Variance Ratio"})
    
    vr_df = vr_df[["K-Period", "Variance Ratio", "P-Value"]]
    vr_df = vr_df.style.apply(highlight, pval_thres = pval_thres, axis = 1)
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

def show_historical_prices_plot(y_test_real, y_test_pred, dates):
    # real = pd.DataFrame(y_test_real,columns=["real"])
    # pred = pd.DataFrame(y_test_pred,columns=["pred"])
    # plot_frame = pd.concat([real,pred],axis=1)
    # plot_frame = pd.concat([plot_frame, dates],axis=1)
    # plot_frame["Date"] = pd.to_datetime(plot_frame['Date'])

    # #bokeh plot 
    # source = ColumnDataSource(plot_frame)
    # p = figure(x_axis_type='datetime')
    
    # p.line(x='Date', y='real', line_width=2, source=source, legend='real prices',color=Spectral3[2])
    # p.line(x='Date', y='pred', line_width=2, source=source, legend='predicted prices')
    # p.yaxis.axis_label = 'Predicted/Real Prices'
    
    # # hover tool tips 
    # hover = HoverTool(
    # tooltips=[
    # ("Date", "@Date{%F}"),
    #     ("Real Price","@real"),
    # ("Predicted Price","@pred")],
    # formatters = {"@Date":"datetime"})

    # hover.mode = 'vline'

    # p.add_tools(hover)
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 2, 4, 5]
    p = figure( title='simple line example', x_axis_label='x',y_axis_label='y')
    p.line(x, y, legend='Trend', line_width=2)
    
    return p


def show_future_expected_returns(model, df_ticker, ticker): 
    expected_ret = get_expected_returns_from_lstm(model, df_ticker, ticker)
    #print(expected_ret)
    ret_form = str(round(expected_ret * 100, 1)) + "%"
    return ret_form

        