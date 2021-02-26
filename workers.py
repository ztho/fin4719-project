import streamlit as st

import numpy as np 
import pandas as pd 

import data_loader as data 
from utils import *
import user_state as u

from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import ColumnDataSource
# from bokeh.models.tools import HoverTool
# from bokeh.layouts import gridplot
from bokeh.palettes import Spectral3


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

def show_historical_prices_plot(df_ticker):
    d = df_ticker.index.max()
    yrs_to_look = 1
    try:
        d = d.replace(year = d.year - yrs_to_look)
    except ValueError:
        d = d + (date(d.year - yrs_to_look, 1, 1) - date(d.year, 1, 1))
    
    try:
        df_ticker = df_ticker.loc[df_ticker.index >= d]
    except ValueError:
        df_ticker = df_ticker

    p = figure(x_axis_type='datetime', height = 300, tools = "reset, save")
    p.sizing_mode = "scale_width"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.yaxis.minor_tick_line_color = None 
    # p.ygrid.band_fill_color = 'cornflowerblue'
    # p.ygrid.band_fill_alpha = 0.1
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    
    p.line(np.array(df_ticker.index, dtype=np.datetime64), df_ticker[df_ticker.columns[0]], color='darkslateblue')
    
    p.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
    return p

def show_model_performance_plot(hist_prices, y_test_pred, y_test_real, dates, split_frac = 0.95): 
    n = int(split_frac * hist_prices.shape[0]) 
    dates = hist_prices[n - 1: ].reset_index().iloc[:, [0]]

    #get recent dates 
    real = pd.DataFrame(y_test_real,columns=["real"])
    pred = pd.DataFrame(y_test_pred,columns=["pred"])
    plot_frame = pd.concat([real,pred],axis=1)
    plot_frame = pd.concat([plot_frame,dates],axis=1)
    plot_frame["Date"] = pd.to_datetime(plot_frame['Date'])

    source = ColumnDataSource(plot_frame)
    p = figure(x_axis_type='datetime', height = 300, tools = "reset, save")
    p.sizing_mode = "scale_width"
    p.grid.grid_line_alpha = 0
    p.yaxis.minor_tick_line_color = None 
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")

    p.line(x='Date', y='real', line_width=2, source=source, legend_label = 'Real prices',color=Spectral3[2])
    p.line(x='Date', y='pred', line_width=2, source=source, legend_label = 'Predicted prices')
    p.yaxis.axis_label = 'Prices'
    
    # hover tool tips 
    hover = HoverTool(
    tooltips=[
    ("Date", "@Date{%F}"),
        ("Real Price","@real"),
    ("Predicted Price","@pred")],
    formatters = {"@Date":"datetime"})

    hover.mode = 'vline'

    p.add_tools(hover)

    return p


def show_future_expected_returns(model, df_ticker, ticker, lookback_period = 90, data_frequency = 252, annualize = True, days_forward = None): 
    expected_ret = get_expected_returns_from_lstm(model, df_ticker, ticker,
                                                 lookback_period = lookback_period, 
                                                 data_frequency = data_frequency,
                                                 annualize = annualize,
                                                 days_forward= days_forward)
    #print(expected_ret)
    ret_form = str(round(expected_ret * 100, 1)) + "%"
    return ret_form

def show_future_expected_vols(pred_prices, days_forward):
    vols = pd.DataFrame(pred_prices)[:days_forward].pct_change().std()[0]
    vols_form = str(round(vols *100, 2)) + "%"
    return vols_form

def show_num_days_predicted(pred_prices, days_forward):
    if (len(pred_prices) < days_forward):
        return "- " + str(len(pred_prices)) + " Days (Max)" 
    return "- " + str(len(pred_prices)) + " Days"

def exec_get_black_litterman_optimal_weights(ticker_list, hist_prices, rel_perf_data):
    if len(ticker_list) >= 2: 
        views_matrix = get_model_views_matrix_arc(ticker_list, rel_perf_data)
        print(views_matrix)
        views, links = get_model_views_and_link_matrices(views_matrix)
        
        gspc = data.get_snp()
        rf = data.get_rf_rate()

        hist_prices_sel = hist_prices.filter(ticker_list)
        covM = get_covariance_matrix(get_returns_from_prices(hist_prices_sel))

        hist_prices_sel_bench = hist_prices_sel.dropna().merge(rf, left_index = True, right_index = True) \
                                                        .merge(gspc, left_index = True, right_index = True) \
                                                        .astype(np.float64)


        opt_weights = get_black_litterman_optimization(hist_prices_sel_bench, covM, "^GSPC", "rf", views, links, tau = 1, allow_short_selling = False)
        # opt_weights = pd.DataFrame(list(opt_weights), index = ticker_list)
        return opt_weights
    else:
        return None

def show_optimal_weights(opt_weights, ticker_list):
    if opt_weights is not None and len(opt_weights) != 0:
        return pd.DataFrame(opt_weights, index = ticker_list, columns = ["Recommended Weight"]) \
            .style.format('{:.2%}')
    else:
        return None
                        
def show_portfolio_returns(opt_weights, hist_prices, ticker_list):
    if opt_weights is None: return None
    if (len(opt_weights) >= 2):
        hist_ret = get_returns_from_prices(hist_prices).filter(ticker_list)
        returns = get_annualized_returns(hist_ret, 252)
        ret =  calc_portfolio_return(opt_weights, returns)
        ret_form = str(round(ret * 100, 1)) + "%"

        return ret_form
    else:
        return "Not Applicable"

def show_portfolio_vols(opt_weights, hist_prices, ticker_list):
    if opt_weights is None: return None
    if (len(opt_weights) >= 2):
        hist_ret = get_returns_from_prices(hist_prices.filter(ticker_list))
        covM = get_covariance_matrix(hist_ret)
        vols = calc_portfolio_vol(opt_weights, covM)
        vols_form = str(round(vols * 100, 1)) + "%"
        return vols_form
    else:
        return "Not Applicable"

def exec_run_cppi(opt_weights, hist_data, ticker_list):
    if opt_weights is not None and len(opt_weights) != 0:
        hist_data_sel = hist_data.filter(ticker_list)
        print(hist_data_sel.shape[1])
        print(len(opt_weights))
        sim, hist = run_cppi(opt_weights, hist_data_sel, m = 2, floor = 0.5, rf = .03, rebal_freq = 100000)
        return sim 

def exec_simulate_gbm(sim, days_sim, init_cap, rf):
    if sim is not None:
        sigma = sim.pct_change().dropna().values.std()
        sim_gbm = simulate_gbm(T = days_sim, S_0 = init_cap, rf = rf, sig = sigma, M = days_sim, num_sim = 1000)
        gbm_res = get_sim_results_stats(sim_gbm)
        var_risk = gbm_res.filter(['end_asset_value']).quantile(.05).values[0]
        return var_risk

def show_VaR(sim, days_sim, init_cap, rf):
    if sim is not None:
        return exec_simulate_gbm(sim, days_sim, init_cap, rf)