import streamlit as st

import numpy as np 
import pandas as pd 
from datetime import timedelta
import data_loader as data 
from utils import *
import user_state as u
from datetime import date

from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.plotting import figure, reset_output
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool
from bokeh.palettes import Spectral3

curdoc().theme ='dark_minimal'

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
    curdoc().theme = "dark_minimal"
    p = figure(x_axis_type='datetime', height = 250, tools = "reset, save")
    p.sizing_mode = "scale_width"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.yaxis.minor_tick_line_color = None 
    # p.ygrid.band_fill_color = 'cornflowerblue'
    # p.ygrid.band_fill_alpha = 0.1
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    
    p.line(np.array(df_ticker.index, dtype=np.datetime64), df_ticker[df_ticker.columns[0]], line_width = 1.5,color='#7564ff')
    
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
    p = figure(x_axis_type='datetime', height = 200, tools = "reset, save")
    p.sizing_mode = "scale_width"
    p.grid.grid_line_alpha = 0
    p.yaxis.minor_tick_line_color = None 
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")

    p.line(x='Date', y='real', line_width=1, source=source, legend_label = 'Real prices',color="#7564ff")
    p.line(x='Date', y='pred', line_width=1, source=source, legend_label = 'Predicted prices', color = "#1ae0ff")
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


# def show_future_expected_returns(model, df_ticker, ticker, lookback_period = 90, data_frequency = 252, annualize = True, days_forward = None): 
#     expected_ret = get_expected_returns_from_lstm(model, df_ticker, ticker,
#                                                  lookback_period = lookback_period, 
#                                                  data_frequency = data_frequency,
#                                                  annualize = annualize,
#                                                  days_forward= days_forward)
#     #print(expected_ret)
#     ret_form = str(round(expected_ret * 100, 1)) + "%"
#     return ret_form
def show_future_expected_returns(pred_prices, days_forward):
    if days_forward > len(pred_prices):
        days_forward = len(pred_prices)
    prices = pd.DataFrame(pred_prices)
    start_price = prices.iloc[0].values[0]
    end_price = prices.iloc[days_forward - 1].values[0]

    exp_ret = (end_price - start_price)/start_price
    ret_form = str(round(exp_ret * 100, 1)) + "%"
    return ret_form
    

def show_future_expected_vols(pred_prices, days_forward):
    vols = pd.DataFrame(pred_prices)[:days_forward].pct_change().std()[0]
    vols_form = str(round(vols *100, 2)) + "%"
    return vols_form

def show_num_days_predicted(pred_prices, days_forward):
    if (len(pred_prices) < days_forward):
        return "- " + str(len(pred_prices)) + " Days (Max)" 
    return "- " + str(len(pred_prices)) + " Days"

def show_predicted_prices_plot(pred_prices, hist_data):
    num_days = len(pred_prices)
    last_day = hist_data.index.max() 

    df_pred_prices = pd.DataFrame(pred_prices, columns = ["pred_prices"])
    dates = [last_day + timedelta(days = i) for i in range(num_days)]
    df_pred_prices.index = dates

    pred_plot = figure(x_axis_type='datetime', height = 200, tools = "")
    pred_plot.sizing_mode = "scale_width"
    pred_plot.grid.grid_line_alpha = 0
    pred_plot.yaxis.minor_tick_line_color = None 
    pred_plot.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    
    pred_plot.line(np.array(df_pred_prices.index, dtype=np.datetime64), df_pred_prices[df_pred_prices.columns[0]], color='#1ae0ff')
    
    pred_plot.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
    return pred_plot


def exec_get_black_litterman_optimal_weights(ticker_list, hist_prices, rel_perf_data):
    if len(ticker_list) >= 2: 
        views_matrix = get_model_views_matrix_arc(ticker_list, rel_perf_data)
        # print(views_matrix)
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

def show_portfolio_returns2(sim):
    if sim is not None:
        rets = sim.pct_change().dropna().values.mean()
        rets_form = str(round(rets * 100, 5)) + "%"
        return rets_form
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

def exec_run_cppi(opt_weights, hist_data, ticker_list, months_back = None):
    if opt_weights is not None and len(opt_weights) != 0:
        hist_data_sel = hist_data.filter(ticker_list)
        # print(hist_data_sel.shape[1])
        # print(len(opt_weights))
        sim, hist = run_cppi(opt_weights, hist_data_sel, m = 2, floor = 0.5, rf = .03, rebal_freq = 100000, months_back = months_back)
        return sim 

def exec_simulate_gbm(sim, days_sim, init_cap, rf, freq = 252, risk_level =.05):
    if sim is not None:
        sigma = sim.pct_change().dropna().values.std()
        # print("Sigma " + str (sigma))
        sim_gbm = simulate_gbm(T = days_sim, S_0 = init_cap, rf = rf/freq, sig = sigma, M = days_sim, num_sim = 100)
        gbm_res = get_sim_results_stats(sim_gbm)
        var_risk = gbm_res.filter(['net_asset_change']).quantile(risk_level).values[0]
        return sim_gbm, var_risk
    return None, None

def show_formatted_VaR(var_risk):
    if var_risk is not None:
        var_form =  str(round(var_risk * 100, 2)) + "%"
        return var_form
    return "Not Applicable"

def show_is_profitable(gbm_sim):
    if gbm_sim is not None:
        gbm_res = get_sim_results_stats(gbm_sim)
        num_profit = len(gbm_res[gbm_res['is_profitable'] == True])
        # print(num_profit)
        perc = str(round((num_profit / len(gbm_res)) * 100, 3)) + "%"
        return perc
    else:
        return "Not Applicable"


def show_portfolio_backtest_plot(sim):

    p = figure(x_axis_type='datetime', height = 200, tools = "reset, save")
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Indicative Value'
    p.ygrid.band_fill_color = None
    p.ygrid.band_fill_alpha = 0
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0") 
    
    p.line(np.array(sim.index, dtype=np.datetime64), sim['Portfolio Value'], line_width = 2, color='#7564ff')
    p.sizing_mode = "scale_width"
    p.yaxis.minor_tick_line_color = None 
    p.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
    return p

def show_portfolio_future_plot(gbm_sim, init_cap, days_sim, hist_data):
    if gbm_sim is not None:

        last_day = hist_data.index.max() 
        dates = [last_day + timedelta(days = i) for i in range(days_sim)]

        sim_res = get_sim_results_stats(gbm_sim)
        
        bottom = sim_res.filter(['net_asset_change']).quantile(.05) / len(gbm_sim) #arithmetic average daily returns
        middle = sim_res.filter(['net_asset_change']).quantile(.5) / len(gbm_sim)
        top = sim_res.filter(['net_asset_change']).quantile(.95) / len(gbm_sim)

        # print(bottom)

        bottom_ind_value = init_cap * np.cumprod([1 + bottom] * days_sim)
        middle_ind_value = init_cap * np.cumprod([1 + middle] * days_sim)
        top_ind_value = init_cap * np.cumprod([1 + top] * days_sim)

        # ind_value = pd.DataFrame(data = {"bottom": bottom_ind_value, 
        #                                 "middle": middle_ind_value, 
        #                                 "top": top_ind_value})

        plot_proj = figure(x_axis_type='datetime', height = 250, tools = "reset, save")
        plot_proj.sizing_mode = "scale_width"
        plot_proj.grid.grid_line_alpha = 0
        plot_proj.xaxis.axis_label = 'Date'
        plot_proj.yaxis.axis_label = 'Indicative Value'
        plot_proj.ygrid.band_fill_color = None
        plot_proj.ygrid.band_fill_alpha = 0
        plot_proj.yaxis.formatter = NumeralTickFormatter(format="$0,0")
        plot_proj.xaxis.minor_tick_line_color = None

        plot_proj.line(dates, bottom_ind_value, color='#006565',
                        legend_label = '5th Percentile', line_width = 1.5)
        plot_proj.line(dates, middle_ind_value, color='#008c8c',
                        legend_label = '50th Percentile', line_width = 1.5)
        plot_proj.line(dates, top_ind_value, color='#00eeee',
                        legend_label = '95% Percentile', line_width = 1.5)

        hover = HoverTool(tooltips=[('Date','$x{%F}'),
                                ('Indicative Value', '$$y{0,0.00}')])
        hover.mode = 'vline'

        plot_proj.add_tools(hover) 

        
        return plot_proj 

