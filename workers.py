####
# Worker module renders frontend objects, calling functions from utils module. Some simple calculations done too 
####

import streamlit as st
import numpy as np 
import pandas as pd 
from datetime import timedelta
from datetime import date

# Import custom supporting packages
import data_loader as data 
from utils import *
import user_state as u # simple state management

# Import graphing packages
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.plotting import figure, reset_output
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool
from bokeh.palettes import Spectral3

# Change Bokeh color theme
curdoc().theme ='dark_minimal'

def show_vr_stats(data, price, k_list, var_form, pval_thres = .05):
    """
    Returns a formatted pd.DataFrame which highlights all rows whose p-value is lower than a prespecified value pval_thres

    :param data: pd.DataFrame - time series in dataframe
    :param price: str - the name of the index in string 
    :param k_list: list - the list of k-period return value in integer
    :param var_form: int - specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    :param pval_thres: float - specified significance level to reject the null hypothesis

    :returns: pd.DataFrame
    """
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
    vr_df = vr_df.convert_dtypes() # get int for k-period
    vr_df = vr_df.style.apply(highlight, pval_thres = pval_thres, axis = 1)
    return vr_df

def show_recent_returns(df_ticker, ticker, num_days):
    """
    Function returns formatted past n-days returns 
    
    :param data:pd.DataFrame -  time series in dataframe 
    :param ticker: str - ticker symbol in string. Ticker symbol must exist in data columns 
    :param num_days: int - number of days to look back on in int (eg num_days = 7 means to look at the past 7 days mean returns)
    
    :returns: str - formatted returns from utils.get_recent_returns()
    """
    if (num_days <= 2): return "Require More Days, Minimum 3"
    ret = get_recent_returns(df_ticker, ticker, num_days)
    ret_form = str(round(ret * 100, 1)) + "%"
    return ret_form

def show_recent_vols(df_ticker, ticker, num_days):
    """
    Function returns formatted past n-days volatility
    
    :param data: time series in dataframe 
    :param ticker: ticker symbol in string. Ticker symbol must exist in data columns 
    :param num_days: number of days to look back on in int (eg num_days = 7 means to look at the past 7 days volatility)
    
    :returns: str - formatted volatility from utils.get_recent_vol()
    """
    vols = get_recent_vol(df_ticker, ticker, num_days)
    vols_form = str(round(vols *100, 2)) + "%"
    return vols_form 

def show_if_random_walk(df_ticker, ticker, k_list, var_form, p_val):
    """
    Function returns if any of the periods violate the null hypothesis in Lo and McKinley's Variance Ratio Test
    
    :param data: pd.DataFrame - time series in dataframe
    :param price: str - the name of the index in string 
    :param k_list: list - the list of k-period return value in integer
    :param var_form: specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    :param p_val: float - specified significance level to reject the null hypothesis
    
    :returns: str - "Yes" if any k-value has p-value lower than prespecified p_val threshold, "No" otherwise
    """
    vr_stats = test_multiple_periods(df_ticker, ticker, k_list, var_form)
    is_random_walk = False if all(p > p_val for p in vr_stats['pval'].values) else True
    return "Yes" if is_random_walk else "No"

def show_historical_prices_plot(df_ticker):
    """
    Function returns the Bokeh plot of historical prices 

    :param data: pd.DataFrame - time series in dataframe

    :returns: Bokeh.line - Bokeh Object containing the plot
    """
    d = df_ticker.index.max()
    yrs_to_look = 2
    try:
        d = d.replace(year = d.year - yrs_to_look)
    except ValueError:
        d = d + (date(d.year - yrs_to_look, 1, 1) - date(d.year, 1, 1))
    
    try:
        df_ticker = df_ticker.loc[df_ticker.index >= d]
    except ValueError:
        df_ticker = df_ticker
    
    curdoc().theme = "dark_minimal"
    
    p = figure(x_axis_type='datetime',
               height = 250, 
               tools = "reset, save, wheel_zoom, pan", 
               toolbar_location = "right")
    p.sizing_mode = "scale_width"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.yaxis.minor_tick_line_color = None 
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    
    p.line(np.array(df_ticker.index, dtype=np.datetime64),
                    df_ticker[df_ticker.columns[0]], 
                    line_width = 1.5, 
                    color='#7564ff',
                    legend_label = df_ticker.columns.values[0] + " Price")
    
    p.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
    p.legend.location = "top_left"
    return p

def show_model_performance_plot(hist_prices, y_test_pred, y_test_real, dates, split_frac = 0.95): 
    """
    Function returns the model predicted prices over the test set in LSTM model 

    :param hist_prices: pd.DataFrame - Historical Prices of the stock
    :param y_test_pred: np.array - predicted prices from LSTM model returned by utils.test_LSTM_model 
    :param y_test_real: np.array - Empirical prices over the same time period returned by utils.test_LSTM_model 
    :param dates: np.array - dates over which the model was tested on 
    :param split_frac: np.float64 - the fraction used to do the train test split in utils.get_train_test_data()

    :returns: Bokeh.line - Bokeh plot with 2 lines - predicted prices and real prices 
    """
    n = int(split_frac * hist_prices.shape[0]) 
    dates = hist_prices[n - 1: ].reset_index().iloc[:, [0]]

    #get recent dates 
    real = pd.DataFrame(y_test_real,columns=["real"])
    pred = pd.DataFrame(y_test_pred,columns=["pred"])
    plot_frame = pd.concat([real,pred],axis=1)
    plot_frame = pd.concat([plot_frame,dates],axis=1)
    plot_frame["Date"] = pd.to_datetime(plot_frame['Date'])

    source = ColumnDataSource(plot_frame)
    p = figure(x_axis_type='datetime',
               height = 200, 
               tools = "reset, save, wheel_zoom, pan"
               )
    p.sizing_mode = "scale_width"
    
    p.grid.grid_line_alpha = 0
    p.yaxis.minor_tick_line_color = None 
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")

    p.line(x='Date', y='real', line_width=1, source=source, legend_label = 'Real Prices',color="#7564ff")
    p.line(x='Date', y='pred', line_width=1, source=source, legend_label = 'Predicted Prices', color = "#1ae0ff")
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
    p.legend.location = "top_left"
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
    """
    Function returns the expected returns as predicted by LSTM model

    :param pred_prices: np.array - predicted prices from utils.predict_prices() 
    :param days_forward: int - number of days forward to project prices 

    :returns: str - formatted expected returns 
    """
    if (days_forward <= 2): return "Require At Least 3 Days" # data validation

    if days_forward > len(pred_prices):
        days_forward = len(pred_prices)
    prices = pd.DataFrame(pred_prices)
    start_price = prices.iloc[0].values[0]
    end_price = prices.iloc[days_forward - 1].values[0]

    exp_ret = (end_price - start_price)/start_price
    ret_form = str(round(exp_ret * 100, 1)) + "%"
    return ret_form
    

def show_future_expected_vols(pred_prices, days_forward):
    """
    Function returns the expected volatility as predicted by LSTM model

    :param pred_prices: np.array - predicted prices from utils.predict_prices() 
    :param days_forward: int - number of days forward to project prices 

    :returns: str - formatted expected volatilities 
    """
    if (days_forward <= 2): return "Require At Least 3 Day" # data validation

    vols = pd.DataFrame(pred_prices)[:days_forward].pct_change().std()[0]
    vols_form = str(round(vols *100, 2)) + "%"
    return vols_form

def show_num_days_predicted(pred_prices, days_forward):
    """
    Function returns number of days predicted. Returns maximum number of days predicted by LSTM model if days_forward
    exceed the nunmber of days predicted in pred_prices.

    :param pred_prices: np.array - predicted prices from utils.predict_prices() 
    :param days_forward: int - number of days forward to project prices 

    :returns: str - the number of days predicted
    """
    if days_forward <= 0: return ""
    if (len(pred_prices) < days_forward):
        return "- " + str(len(pred_prices)) + " Days (Max)" 
    return "- " + str(len(pred_prices)) + " Days"

def show_predicted_prices_plot(pred_prices, hist_data, days_forward):
    """
    Function returns Bokeh plot of predicted future prices as returned from utils.predict_prices()

    :param pred_prices: np.array - predicted prices from utils.predict_prices()
    :param hist_data: pd.DataFrame - historical stock data 
    :param days_forward: int - number of days forward to project prices 

    :returns: Bokeh.line - Bokeh Object with predicted future prices
    """

    if (days_forward <= 0): return None

    num_days = len(pred_prices)
    last_day = hist_data.index.max() 

    df_pred_prices = pd.DataFrame(pred_prices, columns = ["pred_prices"])
    dates = [last_day + timedelta(days = i) for i in range(num_days)]
    df_pred_prices.index = dates

    p = figure(x_axis_type='datetime', 
                       height = 200, 
                       tools = "reset, save, wheel_zoom, pan")
    p.sizing_mode = "scale_width"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.grid.grid_line_alpha = 0
    p.yaxis.minor_tick_line_color = None 
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    
    p.line(np.array(df_pred_prices.index, dtype=np.datetime64), 
                   df_pred_prices[df_pred_prices.columns[0]], 
                   color='#1ae0ff',
                   legend_label = "Future Predicted Prices")
    p.legend.location = "top_left"

    hover = HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode = "vline") #display a tooltip whenever the cursor is vertically in line with a glyph

    p.add_tools(hover)
    
    return p


def exec_get_black_litterman_optimal_weights(ticker_list, hist_prices, rel_perf_data):
    """
    Executes the utils.get_black_litterman_optimization function 

    :param ticker_list: list - list of tickers as string. Found in user_state.tar_stocks
    :param hist_prices: pd.DataFrame which includes all tickers, benchmark and the risk free rate 
    :param rel_perf_data: np.darry - 2D array containing the relative performance of each stock against every other stock
                                    returned by utils.get_model_relative_views()
    
    :returns: np.darray The weight vector of the allocation
    """

    # require minimum 2 stocks for allocation
    if len(ticker_list) >= 2: 
        # Get views matrix from archived source. utils.get_model_views_matrix is too expensive to run on Heroku
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
    """
    Function returns formatted pd.DataFrame of the optimal weights from the black litterman model called in utils.get_black_litterman_optimization

    :param opt_weights: np.array - optimal weights from utils.get_black_litterman_optimization() 
    :param ticker_list: list - list of tickers selected for optimization

    :returns: pd.DataFrame - formatted DataFrame with optimal weights.
    """
    if opt_weights is not None and len(opt_weights) != 0:
        return pd.DataFrame(opt_weights, index = ticker_list, columns = ["Recommended Weight"]) \
            .style.format('{:.2%}')
    else:
        return None
                        
def show_portfolio_returns(opt_weights, hist_prices, ticker_list):
    """
    Returns formatted historical returns from proposed portfolio 

    :param opt_weights: np.array - optimal weights from utils.get_black_litterman_optimization() 
    :param hist_prices: pd.DataFrame - DataFrame containing historical prices 
    :param ticker_list: list - list of tickers selected for optimization

    :returns: str - formatted historical returns
    """
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
    """
    :param opt_weights: np.array - optimal weights from utils.get_black_litterman_optimization() 
    :param hist_prices: pd.DataFrame - DataFrame containing historical prices 
    :param ticker_list: list - list of tickers selected for optimization

    :returns: str - formatted historical volatilities
    """
    if opt_weights is None: return "Not Applicable"
    if (len(opt_weights) >= 2):
        hist_ret = get_returns_from_prices(hist_prices.filter(ticker_list))
        covM = get_covariance_matrix(hist_ret)
        vols = calc_portfolio_vol(opt_weights, covM)
        vols_form = str(round(vols * 100, 1)) + "%"
        return vols_form
    else:
        return "Not Applicable"

def exec_run_cppi(opt_weights, hist_data, ticker_list, init_cap = 1000, months_back = None):
    """
    Exectues utils.run_cppi() 
    
    :param opt_weights: np.array - optimal weights from utils.get_black_litterman_optimization() 
    :param hist_prices: pd.DataFrame - DataFrame containing historical prices 
    :param ticker_list: list - list of tickers selected for optimization
    :param init_cap: np.float64 - initial capital 
    :param months_back: int - number of months back to start the backtesting 

    :returns: pd.DataFrame - The backtested indicative value
    """
    if opt_weights is not None and len(opt_weights) != 0:
        hist_data_sel = hist_data.filter(ticker_list)
        # print(hist_data_sel.shape[1])
        # print(len(opt_weights))
        sim, hist = run_cppi(opt_weights, hist_data_sel, m = 2, init_capital = init_cap, floor = 0.5, rf = .03, rebal_freq = 100000, months_back = months_back)
        return sim 

def exec_simulate_gbm(sim, days_sim, init_cap, rf, freq = 252, risk_level =.05):
    """
    Executes geometric brownian motion simulation as in utils.simulate_gbm() 

    :param sim: pd.DataFrame - simulation results returned by utils.run_cppi() 
    :param days_sim: int - number of days into the future to simulate
    :param init_cap: np.float64 - amount of starting capital 
    :param rf: np.float - risk free rate, annual
    :param freq: int - number of days to scale rf rate (eg. if doing timestep = day, scale by 252)
    :risk_level: np.float - The risk level for VaR

    :returns: (pd.DataFrame, np.float64) - DataFrame of simulation results, VaR value
    """
    if days_sim <= 1: return None, None
    if sim is not None:
        sigma = sim.pct_change().dropna().values.std()
        # print("Sigma " + str (sigma))
        sim_gbm = simulate_gbm(T = days_sim, S_0 = init_cap, rf = rf/freq, sig = sigma, M = days_sim, num_sim = 100)
        gbm_res = get_sim_results_stats(sim_gbm)
        var_risk = gbm_res.filter(['net_asset_change']).quantile(risk_level).values[0]
        return sim_gbm, var_risk
    return None, None

def show_formatted_VaR(var_risk):
    """
    Returns formatted VaR 

    :param var_risk: np.float64 Value-at-risk 
    
    :returns: str - formatted VaR
    """
    if var_risk is not None:
        var_form =  str(round(var_risk * 100, 2)) + "%"
        return var_form
    return "Not Applicable"

def show_is_profitable(gbm_sim):
    """
    Returns the % of profitable simulations 

    :param gbm_sim: pd.DataFrame - simulation results from utils.simulate_gbm() 

    :returns: str - formatted % of profitable simulations 
    """
    if gbm_sim is not None:
        gbm_res = get_sim_results_stats(gbm_sim)
        num_profit = len(gbm_res[gbm_res['is_profitable'] == True])
        # print(num_profit)
        perc = str(round((num_profit / len(gbm_res)) * 100, 3)) + "%"
        return perc
    else:
        return "Not Applicable"


def show_portfolio_backtest_plot(sim):
    """
    Returns the backtested portfolio performance 

    :param sim: pd.DataFrame - simulation results from utils.run_cppi() 

    :returns: Bokeh.line - the backtested portfolio performance 
    """
    p = figure(x_axis_type='datetime', height = 200, tools =  "reset, save, wheel_zoom, pan")
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Indicative Value'
    p.ygrid.band_fill_color = None
    p.ygrid.band_fill_alpha = 0
    p.yaxis.formatter = NumeralTickFormatter(format="$0,0") 
    
    p.line(np.array(sim.index, dtype=np.datetime64), 
           sim['Portfolio Value'], 
           line_width = 2, 
           color='#7564ff',
           legend_label = "Historical Performance")
    p.sizing_mode = "scale_width"
    p.yaxis.minor_tick_line_color = None 
    p.add_tools(HoverTool(tooltips=[('Date', '$x{%F}'),
                                    ('Price', '$$y{0,0.00}')],
                        formatters={'$x': 'datetime'},  #using 'datetime' formatter for 'Date' field
                        mode='vline')) #display a tooltip whenever the cursor is vertically in line with a glyph
    
    p.legend.location = "top_left"
    return p

def show_portfolio_future_plot(gbm_sim, init_cap, days_sim, hist_data):
    """
    Returns the plot of possible future projections from gbm_sim, at the 5%, 50%, 95% confidence

    :param gbm_sim: pd.DataFrame - simulation results from utils.simulate_gbm() 
    :param init_cap: np.float64 - initial capital 
    :param days_sim: int - number of days to simulate 
    :param hist_data: pd.DataFrame - the historical prices (to get the last day of data)

    :returns: Bokeh.line - 3 Lines outlining the % chance of getting above certain values
    """
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

        ind_value = pd.DataFrame(data = {"bottom_ind_value": bottom_ind_value, 
                                        "middle_ind_value": middle_ind_value, 
                                        "top_ind_value": top_ind_value})
        ind_value['dates'] = dates
        source = ColumnDataSource(ind_value)

        plot_proj = figure(x_axis_type='datetime', height = 250, tools =  "reset, save, wheel_zoom, pan")
        plot_proj.sizing_mode = "scale_width"
        plot_proj.grid.grid_line_alpha = 0
        plot_proj.xaxis.axis_label = 'Date'
        plot_proj.yaxis.axis_label = 'Indicative Value'
        plot_proj.ygrid.band_fill_color = None
        plot_proj.ygrid.band_fill_alpha = 0
        plot_proj.yaxis.formatter = NumeralTickFormatter(format="$0,0")
        plot_proj.xaxis.minor_tick_line_color = None

        plot_proj.line(x ="dates",y= "bottom_ind_value", color='#006565',source=source,
                        legend_label = '5th Percentile', line_width = 1.5)
        r1 = plot_proj.line(x="dates", y="middle_ind_value", color='#008c8c',source=source,
                        legend_label = '50th Percentile', line_width = 1.5)
        plot_proj.line(x="dates", y="top_ind_value", color='#00eeee',source=source,
                        legend_label = '95% Percentile', line_width = 1.5)

        hover = HoverTool(tooltips=[('Date','@dates{%F}'),
                        ("Projected Value, 5% chance of having more than", '$@top_ind_value{0,0.00}'),
                            ("Projected Value, 50% chance of having more than",'$@middle_ind_value{0,0.00}'),
                                ("Projected Value, 95% chance of having more than",'$@bottom_ind_value{0,0.00}')],
                    formatters = {"@dates":"datetime"})
        hover.renderers = [r1]
        hover.mode = 'vline'

        plot_proj.add_tools(hover) 
        
        plot_proj.legend.location = "top_left"
        return plot_proj 

