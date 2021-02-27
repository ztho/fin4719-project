#import all dependencies 
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize 
import scipy.stats as stats
from sklearn.covariance import LedoitWolf, ledoit_wolf
from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm 

from sklearn import preprocessing
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
# from keras import optimizers, callbacks, losses, models
# import tensorflow as tf

# config = tf.compat.v1.ConfigProto( #set not to use gpu
#         device_count = {'GPU': 0}
#     )
# sess = tf.compat.v1.Session(config=config)

# Portfolio Optimization Functions 

def get_returns_from_prices(hist_prices, log_prices = True):
    """
    Function computes the log_prices returns from a DataFrame containing historical prices 

    :param hist_prices: pd.DataFrame containing historical prices 
    :param log_priceS: Boolean - True default to use log prices calculation, else use simple pct_change()

    :returns: pd.DataFrame of returns of stocks
    """
    if log_prices:
        return np.log(1 + hist_prices.pct_change().dropna(how="all"))
    else:
        return hist_prices.pct_change().dropna()

def get_annualized_returns(hist_returns, periods_per_year):
    """
    Function returns array containing annualized returns of individual stocks
    
    :param hist_returns: pd.DataFrame containing historical prices
    :param periods_per_year: int - number of periods per year to compound. (e.g if data is monthly data, then periods per year = 12)
    
    :returns: np.array containing annualized returns of all stocks
    """
    cg = (1 + hist_returns).prod() 
    return cg ** (periods_per_year/hist_returns.count()) - 1

def get_covariance_matrix(hist_returns, use_ledoit_wolf = True, frequency = 252, annualize = True):
    """
    Function returns the covariance matrix as a DataFrame. Uses Ledoit Wolf shrinkage by default

    :param hist_returns: pd.DataFrame containing historical prices
    :param use_ledoit_wolf: Boolean - Default True to use shrinkage, False for sample covariance 
    :param frequency: int - interval of the data in hist_returns (e.g if daily data, use convention 252 trading days)
    :param annualize: Boolean - Default True returns the covariance of yearly returns, False for whichever frequency is specified

    :return: pd.DataFrame - Covariance Square Matrix
    """
    if use_ledoit_wolf:
        cov_lf = ledoit_wolf(hist_returns.dropna())[0]
        output = pd.DataFrame(data = cov_lf, index = hist_returns.columns, columns = hist_returns.columns)
        if annualize:
            return output * frequency 
        else:
            return output
    else:
        if annualize:
            return hist_returns.dropna().cov() * frequency
        else:
            return hist_returns.dropna().cov()

def calc_portfolio_return(weights, returns):
    """
    Function takes the dot product of the weight and return vectors and return the expected returns of the portfolio 

    :param weights: np.array weight vector of the assets (relative weights)
    :param returns: np.array the expected returns of individual assets

    :returns: np.float64 expected return of the portfolio
    """
    return weights @ returns

def calc_portfolio_vol(weights, covM):
    """
    Function takes the dot product of the weight and covariance matrix to get portfolio volatility

    :param weights: np.array weight vector of the assets (relative weights)
    :param covM: pd.DataFrame the covariance matrix from get_covariance_matrix() 

    :returns: np.float64 volatility of the portfolio (standard deviation)
    """
    return (weights.T @ covM  @ weights) ** 0.5

def get_min_vol_weights(target_return, er, cov, max_weight = 1.0):
    """
    Given a target return, return a weight vector

    :param target_return: np.float64 the targeted return desired 
    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets 
    :param max_weight: np.float64 maximum allowable weight individual assets

    :returns: np.array Weight vector of the assets which achieve the target return with the lowest possible volatility
    """
    n = er.shape[0] #figure out the number of assets 
    init_guess = np.repeat(1/n, n) #initial weights for gradient descent
    bounds = ((0.0, 1.0),) * n #tuple of tuples, specifies that the weights of each of the n assets 
    # is between 0 and 1
    #Setting up the constraints for scipy.optimize
    return_is_target = {
        'type': 'eq',
        "args": (er,),
        "fun": lambda w, er: target_return - calc_portfolio_return(w, er) 
        #equalitiy must be equal to 0!!!
    }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    max_exposure = {
        "type": "ineq",
        "fun": lambda w: -w + max_weight
    }
    #(objective function, init_guess)
    results = minimize(calc_portfolio_vol, init_guess,
                args = (cov,), method = "SLSQP",#using quadratic
                options = {"disp": False}, #remove excess info
                constraints = (return_is_target, weights_sum_to_1, max_exposure),
                bounds = bounds 
                )
    return results.x 

def get_max_returns(target_vol, er, cov, max_weight = 1.0):
    """
    Get Max Possible Returns Given Risk (standard deviation). Returns Weights 

    :param target_return: np.float64 the targeted return desired 
    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets 
    :param max_weight: np.float64 maximum allowable weight individual assets

    :returns: np.array Weight vector of the assets which achieve the highest return given a level of volatility
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((.0, 1.0), ) * n
    
    return_is_target = {
        "type": "eq",
        "args": (cov, ),
        "fun": lambda w, cov: target_vol - calc_portfolio_vol(w, cov) 
    }
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    max_exposure = {
        "type": "ineq",
        "fun": lambda w: -w + max_weight
    }   
    def neg_calc_portfolio_return(w, er):
        return - calc_portfolio_return(w, er)
    
    results = minimize(neg_calc_portfolio_return, init_guess,
                args = (er,), method = "SLSQP",#using quadratic
                options = {"disp": False}, #remove excess info
                constraints = (return_is_target, weights_sum_to_1, max_exposure),
                bounds = bounds 
                )
    return results.x 

    
def optimal_weights(n_points, er, cov, max_weight = 1.0):
    """
    Return a list of weights to run the optimizier on to minimize the volatility

    :param n_points: int the number of points desired to simulate
    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets 
    :param max_weight: np.float64 maximum allowable weight individual assets

    :returns: np.array Array of the weight vectors for minimum volatiilty at a given return level
    """
    target_rets = np.linspace(er.min(), er.max(), n_points)
    w = [get_min_vol_weights(target_return, er, cov, max_weight) for target_return in target_rets]
    return w 

def get_max_sharpe_optimal_weights(er, cov, rf = 0.03, max_weight = 1.0, with_ticker = False): 
    """
    Returns a weight vector of multiple assets given a rf, er, cov
    which maximises the Sharpe ratio

    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets 
    :param rf: np.float64 the risk free rate, in decimal places
    :param max_weight: np.float64 maximum allowable weight individual assets
    :param  with_ticker: Boolean Default False to return weight vector, True to return as a pd.DataFrame

    :returns: either a np.array weight vector or pd.DataFrame depending on with_ticker param
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #initial weights for gradient descent
    bounds = ((0.0, 1.0),) * n #tuple of tuples, specifies that the weights of each of the n assets 
    # is between 0 and 1
    #Setting up the constraints for scipy.optimize
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    max_exposure = {
        "type": "ineq",
        "fun": lambda w: -w + max_weight
    }
    def neg_sharpe_ratio(w, rf, er, cov): 
        rets = calc_portfolio_return(w, er)
        vol = calc_portfolio_vol(w, cov)
        return -((rets-rf) / vol)
    
    #(objective function, init_guess)
    #minimize -ve sharpe ratio = maximise sharpe!
    results = minimize(neg_sharpe_ratio, init_guess,
                args = (rf, er, cov, ), method = "SLSQP",#using quadratic
                options = {"disp": False}, #remove excess info
                constraints = (weights_sum_to_1, max_exposure),
                bounds = bounds 
                )
    # print(results.fun)
    if (with_ticker == False):
        return results.x
    else:
        data = {"Ticker": er.index.tolist(), "Weight": results.x}
        output = pd.DataFrame(data, columns = ["Ticker", "Weight"])
        return output
    
def get_max_sharpe_value(er, cov, rf = 0.03, max_weight = 1.0): 
    """
    Returns the maximized Sharpe Ratio value

    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets 
    :param rf: np.float64 the risk free rate, in decimal places
    :param max_weight: np.float64 maximum allowable weight individual assets
    
    :returns: np.float64 the numerical value of the Sharpe Ratio
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #initial weights for gradient descent
    bounds = ((0.0, 1.0),) * n #tuple of tuples, specifies that the weights of each of the n assets 
    # is between 0 and 1
    #Setting up the constraints for scipy.optimize
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
    def neg_sharpe_ratio(w, rf, er, cov): 
        rets = calc_portfolio_return(w, er)
        vol = calc_portfolio_vol(w, cov)
        #print(-((rets-rf) / vol))
        return -((rets-rf) / vol)
    
    #(objective function, init_guess)
    #minimize -ve sharpe ratio = maximise sharpe!
    results = minimize(neg_sharpe_ratio, init_guess,
                args = (rf, er, cov, ), method = "SLSQP",#using quadratic
                options = {"disp": False}, #remove excess info
                constraints = (weights_sum_to_1),
                bounds = bounds 
                )
    return -results.fun    

def plot_efficient_frontier(n_points, er, cov, rf = 0.03, max_weight = 1.0, style = ".-", show_cml = True):
    """
    Plots the multi-asset efficient frontier

    :param n_points: int the number of points desired to simulate
    :param er: np.array expected returns of each asset 
    :param cov: pd.DataFrame covariance matrix of assets
    :param rf: np.float64 the risk free rate, in decimal places 
    :param max_weight: np.float64 maximum allowable weight individual assets
    :param style: str the style type of plotting
    :param show_cml: Boolean Default True to show Capital Market Line, False Otherwise

    :return: matplotlib.axes The Efficient Frontier
    """
    w = optimal_weights(n_points, er, cov)
    rets = [calc_portfolio_return(w, er) for w in w]
    vols = [calc_portfolio_vol(w, cov) for w in w]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    
    ax = ef.plot.line(x = "Volatility", y = "Returns", style = style)
    if show_cml:
        ax.set_xlim(left = 0)
        #add CML 
        w_msr = get_max_sharpe_optimal_weights(er, cov, rf, max_weight)
        r_msr = calc_portfolio_return(w_msr, er)
        cml_x = [0, vol_msr]
        cml_y = [rf, r_msr]
        ax.plot(cml_x, cml_y, color = "green", marker = "o", linestyle = "dashed")
        
    return ax 

# Advanced Portfolio Allocation Algorithms 
def get_regression_res(hist_prices, benchmark_col_name, rf_col_name):
    """
    Function takes in df of historical prices including the rf rates and benchmark prices.
    Returns relevant statistics on market model

    :param hist_prices: pd.DataFrame which includes all tickers, benchmark and the risk free rate
    :param benchmark_col_name: str - the column name of the benchmark in hist_prices
    :param rf_col_name: str - the column name of the risk free rate in hist_prices 

    :returns: pd.DataFrame containing all relevant statistics from single factor model regression
    """
    benchmark_excess = hist_prices.filter([benchmark_col_name]).pct_change().dropna() - hist_prices.filter([rf_col_name])[1:].values
    benchmark_excess_np = np.array(benchmark_excess.values)
    
    tickers = hist_prices.columns[:-2]
    
    out = pd.DataFrame(columns = {"ticker","std_dev", "beta", "alpha", "std_dev_resid"})
    for stock in tickers:
        excess_ret = hist_prices.filter([stock]).pct_change().dropna() - hist_prices.filter([rf_col_name])[1:].values
        X_intercept = sm.add_constant(benchmark_excess_np, prepend=False)
        ols = sm.OLS(excess_ret, X_intercept)
        ols_result = ols.fit()

        out = out.append({"ticker": stock, 
                          "beta": ols_result.params[0], 
                          "beta_p_value": ols_result.pvalues.values[0],
                          "alpha": ols_result.params["const"], 
                          "alpha_p_value": ols_result.pvalues.values[1], 
                          "std_dev": excess_ret.values.std(), 
                          "std_dev_resid": ols_result.resid.std()},
                         ignore_index = True)
    out = out.set_index("ticker", drop = True)
    return out

def get_treynor_black_optimization(hist_prices, benchmark_col_name, rf_col_name):
    """
    Function calculates the optimal weight allocations based on Treynor-Black's Optimization

    :param hist_prices: pd.DataFrame which includes all tickers, benchmark and the risk free rate
    :param benchmark_col_name: str - the column name of the benchmark in hist_prices
    :param rf_col_name: str - the column name of the risk free rate in hist_prices 

    :returns: np.array - array containing the weight vector
    """
    stats = get_regression_res(hist_prices, benchmark_col_name, rf_col_name)
    resid_var = stats['std_dev_resid'] ** 2
    info_ratio = stats['alpha'] / resid_var
    scaled_weights = info_ratio / info_ratio.sum()
    
    port_alpha = scaled_weights @ stats['alpha']
    port_resid_var = resid_var @ scaled_weights
    return scaled_weights

def get_black_litterman_optimization(hist_prices, covM,  
                                     benchmark_col_name, rf_col_name, 
                                     views_matrix, link_matrix, tau = 1, allow_short_selling = True):
    """
    Function finds the optimal weight allocation by Black-Litterman Model. 
    
    :param hist_prices: pd.DataFrame which includes all tickers, benchmark and the risk free rate 
    :param covM: pd.DataFrame which is the covariance matrix of all tickers (not including benchmark and rf), obtained by get_covariance_matrix()
    :param benchmark_col_name: str - the column name of the benchmark in hist_prices
    :param rf_col_name: str - the column name of the risk free rate in hist_prices 
    :param views_matrix: np.array: an 1D array containing the expected relative outperformance of 1 security to another 
    :param link_matrix: np.array: a 2D array which represents which security outperforms which other security 
    :param tau: np.float64 the value of tau, quantifying the uncertainty of views in Black-Litterman Model 
    "param allow_short_selling": Boolean, True if allowing securities to be shorted. False otherwise

    :returns: np.darray The weight vector of the allocation
    """
    # get implied equilibrum excess returns (pi)
    pi = np.array(get_regression_res(hist_prices, benchmark_col_name, rf_col_name).filter(['alpha'])).flatten()

    # get covariance matrix 
    
    ticker_only_returns = get_returns_from_prices(hist_prices[hist_prices.columns[:-2]], log_prices = True)
    #S = np.array(get_covariance_matrix(ticker_only_returns, use_ledoit_wolf = True, frequency = 252, annualize = True)) 
    S = np.array(covM.copy())
    
    # define our views (Q)
    Q = views_matrix

    # define the link matrix of which we are expressing our views(P)
    P = link_matrix

    # get the uncerntainty of our views 
    omega = tau * (P @ S @ P.T)

    # calculate expected return
    try:
        ex1 = np.linalg.inv((np.linalg.inv((tau * S))) + (P.T @ np.linalg.inv(omega) @ P))
        ex2 = np.linalg.inv(tau * S) @ pi + P.T @ np.linalg.inv(omega) @ Q
    except:
        return []

    mu_bl = ex1 @ ex2 # expected excess return under black litterman

    # get envelope portfolio 
    Z = np.linalg.inv(S) @ mu_bl

    denom = Z.sum() 
    
    if allow_short_selling:
        weight_vector = Z / denom
        return weight_vector
    
    else:
        n = ticker_only_returns.shape[1] # get number of assets 
        init_guess = np.repeat(1/n, n)
        bounds = ((.0, 1.0), ) * n
        
        weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
        }
        
        no_short_weight = {
        "type": "ineq",
        "fun": lambda w: w - 0 
        }
        
        def neg_sharpe_ratio(w, er, cov): 
            rets = calc_portfolio_return(w, er)
            vol = calc_portfolio_vol(w, cov)
            return -((rets) / vol)
        
        results = minimize(neg_sharpe_ratio, init_guess,
                args = (mu_bl, S, ), method = "SLSQP",#using quadratic
                options = {"disp": False}, #remove excess info
                constraints = (weights_sum_to_1, no_short_weight),
                bounds = bounds 
                )
        return results.x

# Backtesting Algorithms 
def run_cppi(weight_alloc, 
             hist_data, 
             safe_r= None, 
             m=3, 
             init_capital = 1000.,
             floor = 0.8, 
             rf = 0.03,
             rebal_freq = 6,
             months_back = None):
    """
    Simulate CPPI strategy using historical prices.
    
    :param weight_alloc: np.array of weight allocations, in the order same as the hist_data.columns
    :param hist_data: pd.Dataframe of historical data
    :param safe_r: pd.DataFrame containing historical risk free rates 
    :param m: int - the multiplier value to risk in capital markets 
    :param init_capital: np.float64 - initial amount of capital 
    :param floor: the fraction of the value of the portfolio that is must not fall under
    :param rf: np.float64 the static risk free rate to use if safe_r is not available. Assumes constant rf throughout simulation
    :param rebal_freq: int - how often to rebalance the portfolio 
    :param months_back: int - specify how long ago the backtest should start (eg. 6 means start the backtest 6 months ago)
    :returns: A tuple containing 2 dataframes, sim: the backtested historical portfolio value, 
                                            hist: the historical value of the individual holdings
    """
    
    assert len(weight_alloc) == hist_data.shape[1], "Weight vector has different number of assets compared to historical data"
    
    pct_change = hist_data.pct_change().dropna() + 1
    
    d = pct_change.index.max()
    if months_back is not None:
        try:
            d = d + pd.DateOffset(months = - months_back)
            pct_change = pct_change.loc[pct_change.index >= d]
        except:
            pass 

    # Set up CPPI parameters
    risky_capital = (1-floor) * init_capital * m #capital you are willing to risk
    floor_value = floor * init_capital #this is if it breaks this floor value, it is not the floor amount!
    floor_amount = init_capital - risky_capital
    cap_alloc = weight_alloc * risky_capital
    
    # setup risk-free asset 
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(pct_change)
        safe_r = safe_r.drop(safe_r.columns[1:], axis =1)
        safe_r.values[:] = rf/12 
        
    dates = pct_change.index
    n_steps = len(dates)
    
    port_holdings_history = pd.DataFrame().reindex_like(pct_change)
    port_cum_history = pd.DataFrame().reindex_like(pct_change)
    port_cum_history = port_cum_history.drop(port_cum_history.columns[1:],axis=1).rename(columns={port_cum_history.columns[0] : "Portfolio Value"})
    
    for step in range(n_steps):
        if(step == 0):
            port_holdings_history.iloc[step] = cap_alloc
            #print(port_holdings_history.iloc[step].sum() + floor_value)
            port_cum_history.iloc[step] = port_holdings_history.iloc[step].sum() + floor_amount
            #print(port_cum_history.iloc[step])
        else:
            if (rebal_freq != 0 and step%rebal_freq == 0):
                if (port_cum_history.iloc[step][0] < floor_value):
                    # print("Rebalance")
                    risky_capital = (1-floor) * port_cum_history.iloc[step - 1] * m
                    #print(risky_capital)
                    cap_alloc = weight_alloc * risky_capital
                    floor_amount = port_cum_history.iloc[step][0] - risky_capital
                    port_holdings_history.iloc[step] = cap_alloc
                    port_cum_history.iloc[step] = port_holdings_history.iloc[step].sum() + floor_amount
                else:
                    #print(step)
                    #print(port_cum_history.iloc[step-1])
                    floor_value = max(floor * port_cum_history.iloc[step - 1].sum(),floor_value)
                    #print(floor_value)
                    port_holdings_history.iloc[step] = port_holdings_history.iloc[step-1] * pct_change.iloc[step]
                    port_cum_history.iloc[step] = port_holdings_history.iloc[step].sum() + floor_amount
            else:
                floor_value = max(floor * port_cum_history.iloc[step - 1].sum(), floor_value)
                #print(floor_value)
                port_holdings_history.iloc[step] = port_holdings_history.iloc[step-1] * pct_change.iloc[step]
                port_cum_history.iloc[step] = port_holdings_history.iloc[step].sum() + floor_amount
        #print("Floor Value: " + str(floor_value) + " Port Value: " + str(port_cum_history.iloc[step][0]))
    return port_cum_history, port_holdings_history

# Statistical Calculation Functions After Backtesting Algorithm
def skewness(r): 
    demeaned_r = r - r.mean() 
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r ** 3).mean() 
    return exp/sigma_r**3 

def kurtosis(r): 
    """
    Alternative to scipy.stats.kurtosis()
    But scipy.stat.kurtosis() returns excess Kurtosis
    Computes Kurtosis
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

# Monte Carlo Simulation Algorithms
def simulate_gbm(T = 30, S_0 = 100., rf = .0065, sig = .2, div = 0, 
                M = 30, num_sim = 250000):
    """
    Function accepts a multiple of parameters to return a Monte Carlo Simulation of future prices
    
    :param T: int - Time Steps to end of simulation
    :param S_0: np.float64 - starting value of portfolio/asset
    :param rf: np.float64 - assumed risk free rate 
    :param sig: np.float64 - volatility of portfolio 
    :param M: int - number of time steps to simulate
    :param num_sim: int - number of simulations to run
    
    :returns: pd.DataFrame containing the simulated prices movements
    """
    dt = math.floor(T / M)
    
    #get price movement matrix (vectorized implementation)
    S = S_0 * np.exp(
        np.cumsum((rf - div - 0.5 * sig **2) * dt
        + sig * math.sqrt(dt) * np.random.standard_normal((M + 1, num_sim)), axis = 0))
    
    return pd.DataFrame(S)

def plot_gbm(sim_results, init_price = 100):
    """
    Function accepts output from simulate_gbm and a initial portfolio value and plots the simulated returns as simulated
    
    :param sim_results: pd.DataFrame - output of simulate_gbm
    :param init_price: np.float64 - initial value of portfolio
    """
    ax = sim_results.plot(legend = False, color = "indianred", alpha = 0.5, linewidth = 2, figsize= (12, 5))
    ax.axhline(y= init_price, ls=":", color="black")
    ax.set_title("Stochastic Scenario Analysis")
    ax.set_xlabel("Days")
    ax.set_ylabel("Relative Value")

# Statistics Functions For Monte Carlo Simulations

def get_max_drawdown(r):
    """
    This functions accepts a dataframe of returns and computes the max drawndown, 
    returned as a dataframe
    """
    #Simulate an index with initial capital 1000 
    wealth_index = 1000 * (1 + r).cumprod()
    prev_peaks = wealth_index.cummax() # find current max values 
    drawdown = (wealth_index - prev_peaks) / prev_peaks #get percentage 
    return drawdown

def var_gaussian(r, level = 5, mod = False):
    """
    Returns the Parametric Gaussian VaR of a Series or Dataframe
    If mod = True, then modified VaR is returned via Cornish-Fisher modification
    """
    z = stats.norm.ppf(level / 100) 
    if (mod):
        s = skewness(r) 
        k = kurtosis(r) 
        z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 -5*z)*(s**2)/36)
    return -(r.mean() + z * r.std(ddof = 0))

    
def get_sim_results_stats(sim_results):
    """
    Function accepts results from simulate_gbm and computes important simulation statistics such as max drawdown, percentage of profitable portfolios 
    , ending asset values of the simulation, and the net asset change in value i.e the rate of return of the portfolio
    
    :param sim_results: pd.DataFrame - output of simulate_gbm
    
    :returns: pd.DataFrame - summary statistics in a dataframe
    """
    output = pd.DataFrame()
    output["max_drawdown"] = get_max_drawdown(sim_results.pct_change()).min()
    output["is_profitable"] = sim_results.iloc[0] < sim_results.iloc[len(sim_results) - 1]
    output['end_asset_value'] = sim_results.iloc[len(sim_results) - 1]
    output['net_asset_change'] = (sim_results.iloc[len(sim_results) - 1] - sim_results.iloc[0])/sim_results.iloc[0]
    return output

def output_sim_stats(sim_prices):
    """Function plots the results and gives other statistics"""
    summary = get_sim_results_stats(sim_prices)
    max_draw = round((-100* min(summary['max_drawdown'])),2)
    num_profit = len(summary[summary['is_profitable'] == True])
    pct_num_profit = round(num_profit/len(summary) * 100, 2)
    avg_net_asset_change = round(100 * summary["net_asset_change"].mean(), 2)

    print("The max drawdown in the simulation is " + str(max_draw) + "%")
    print("Number of times it is profitable is " + str(num_profit) + " out of " + str(len(summary)) + " simulations, or " + str(pct_num_profit) + "%") 
    print("On average, the net asset gain is " + str(avg_net_asset_change) + "%")
    print("Distribution of ending asset value as below")
    
    ax = sns.distplot(summary["end_asset_value"])
    ax.set_title("Distribution of Ending Asset Value")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Ending Asset Value")

# Prediction Models
def calc_var_ratio(data, price, k):
    """
    Calculates the variance ratio of a time series given a specified interval k 
    
    :param data: time series in dataframe
    :param price: the name of the index in string 
    :param k: the k-period return value in integer
    
    :returns: variance ratio in float64
    """
    prices = data[price].dropna().to_numpy(dtype = np.float64) #convert prices to floating type
    log_prices = np.log(prices) #get the natural log of prices
    rets = np.diff(log_prices) #get the difference in log prices, i.e X_t - X_(t-1)
    T = len(rets) #get number of data points 
    mu = np.mean(rets) #get avg return 
    
    #get sig1
    var_1 = np.var(rets, ddof = 1, dtype = np.float64) #get varianace
    
    #get sigK
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T) #as defined in Lo and MacKinley
    var_k = (1 / m) * np.sum(np.square(rets_k - k * mu))
    
    #Get variance ratio
    vr = var_k / var_1
    
    return vr

def calc_phi(data, price, k, var_form = 2):
    """
    Returns the phi value as defined in Lo and Mackinley
    
    :param data: time series in dataframe
    :param price: the name of the index in string 
    :param k: the k-period return value in integer
    :param var_form: specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    
    :returns: phi value in float64
    """
    prices = data[price].dropna().to_numpy(dtype = np.float64) #convert prices to floating type
    rets = np.diff(np.log(prices))
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof = 1, dtype = np.float64) #get varianace
    
    def delta(j):
        res = 0 
        for t in range(j + 1, T + 1): 
            t -= 1
            res += np.square((rets[t] - mu) * (rets[t - j] - mu)) 
        return res / ((T - 1) * var_1)**2
    phi = 0
    if(var_form == 2):
        for j in range(1, k):
            phi += (2 * (k - j) / k) ** 2 * delta(j)
    else:
        #print("triggered")
        phi = 2 * ((2 * k) - 1) * (k - 1)/ (3 * k * T)
    return phi

def calc_test_stat(data, price, k, var_form = 2):
    """
    Returns the test statistic value as defined in Lo and MacKinley
    
    :param data: time series in dataframe
    :param price: the name of the index in string 
    :param k: the k-period return value in integer
    :param var_form: specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    
    :returns: t-stat value of vr 
    """
    return (calc_var_ratio(data, price, k) - 1)/ np.sqrt(calc_phi(data, price, k, var_form))

def get_all_vr_stats(data, price, k, var_form = 2):
    """
    :param data: time series in dataframe
    :param price: the name of the index in string 
    :param k: the k-period return value in integer
    :param var_form: specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    
    :returns: tuple containing (variance ratio, t-stat value of vr, p-value, number of observations)
    """
    T = len(data[price].dropna().to_numpy(dtype = np.float64)) - 1
    t_stat = calc_test_stat(data, price, k , var_form)
    pval = stats.t.sf(np.abs(t_stat), T - 1)* 2
    return calc_var_ratio(data, price, k), t_stat, pval, T

def test_multiple_periods(data, price,  k_list = [2,4,8,16,32] , var_form = 2):
    """
    Function calculates the variance ratio and relevant statistics for a range of k-period values 
    
    :param data: time series in dataframe
    :param price: the name of the index in string 
    :param k_list: list - the list of k-period return value in integer
    :param var_form: specify if homoscedastic (1) or heteroscedastic (2). Default 2 in integer 
    
    :returns: pd.DataFrame variance ratio and relevent statistics for each k value
    """
    out = pd.DataFrame()
    for k in k_list:
        vr, stat, pval, T = get_all_vr_stats(data, price, k, var_form)
        out = out.append({'k': int(k), 'vr': vr, "stat": stat, "pval": pval}, ignore_index = True)
    return out

# Functions For Calculating Recent Data
def get_recent_returns(data, ticker, num_days = 7):
    """
    Function returns the past n-days returns 
    
    :param data: time series in dataframe 
    :param ticker: ticker symbol in string. Ticker symbol must exist in data columns 
    :param num_days: number of days to look back on in int (eg num_days = 7 means to look at the past 7 days mean returns)
    
    :returns: np.float64 the mean returns in decimal
    """
    if ticker not in data.columns:
        raise Exception("Ticker not found in DataFrame")
        
    if num_days <= 1: 
        raise Exception("Minimum number of days is 2")
    
    ret = data[ticker].iloc[-num_days: ].pct_change().mean()
    
    if not math.isnan(ret):
        return ret 
    else:
        return data[ticker].pct_change().mean()
    
def get_recent_vol(data, ticker, num_days = 7):
    """
    Function returns the past n-days volatility
    
    :param data: time series in dataframe 
    :param ticker: ticker symbol in string. Ticker symbol must exist in data columns 
    :param num_days: number of days to look back on in int (eg num_days = 7 means to look at the past 7 days volatility)
    
    :returns: np.float64 the mean returns in decimal.
    """
    if ticker not in data.columns:
        raise Exception("Ticker not found in DataFrame")
        
    if num_days <= 2: 
        raise Exception("Minimum number of days is 3")
    
    ret = data[ticker].iloc[-num_days: ].pct_change().std()
    
    if not math.isnan(ret):
        return ret 
    else:
        return data[ticker].pct_change().std()

# Function to calculate technical indicators
def calc_ema(hist_prices, window = 14):
    return hist_prices.ewm(span = window, adjust = False).mean()

def calc_rsi(hist_prices, window = 14, calc_type = "ema"):
    """
    Function returns the exponential moving average 
    Parameters:
        hist_prices - a Dataframe consisting of historical prices. DataFrame to have only 1 index and 1 column 
        window - the number of days to take EMA from
        calc_type - the type of moving average used in calculation. Either "ema" or "sma". Default ema
    Returns:
        Dataframe of values for EMA 
    """
    assert hist_prices.shape[1] == 1, "DataFrame must only have 1 column labelled by its ticker and corresponding price" 
    
    ticker = hist_prices.columns[0]
    #get difference in prices 
    delta = hist_prices.diff()[1:] #remove first entry since NaN
    
    #get ups and downs
    up, down = delta.copy(), delta.copy() 
    up[up < 0] = 0
    down[down > 0] = 0
    
    #Calculate EWMA
    roll_up1 = up.ewm(span = window).mean() 
    roll_down1 = down.abs().ewm(span = window).mean() 
    
    #get RSI based on EWMA
    RS1 = roll_up1 / roll_down1 
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    
    #calculate SMA 
    roll_up2 = up.rolling(window).mean() 
    roll_down2 = down.abs().rolling(window).mean()
    
    #get RSI based on SMA 
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    
    RSI1 = RSI1.rename(columns = {ticker: "rsi"})
    RSI2 = RSI2.rename(columns = {ticker: "rsi"})
    if (calc_type == 'sma'):
        return RSI2
    else:
        return RSI1

def calc_macd(hist_prices):
    """
    Returns the time series of MACD and Signal as a Dataframe given a dataframe of historical prices
    
    :param hist_prices: dataframe of historical prices of 1 asset
    :return: dataframe
    """
    
    assert hist_prices.shape[1] == 1, "DataFrame must only have 1 column labelled by its ticker and corresponding price" 
    
    #macd = calc_ema(hist_prices, window = 12) - calc_ema(hist_prices, window = 26)
    macd = hist_prices.ewm(span = 12, min_periods = 12).mean() - hist_prices.ewm(span = 26, min_periods = 12).mean() 
    ticker = macd.columns[0]
    macd = macd.rename(columns = {ticker: "macd"})
    
    signal = macd.ewm(span = 9, min_periods = 9).mean()
    #signal = calc_ema(hist_prices, 9)
    output = macd.copy() 
    output['macd_signal'] = signal.values
    #macd = macd.join(pd.Series(signal, name = "signal"))
    return output.dropna()

def calc_technical_indicators(hist_prices): 
    """
    Use own functions to calculate technical indicators (RSI and MACD)
    """
    rsi = calc_rsi(hist_prices, 14)
    macd = calc_macd(hist_prices) 
    output = macd.merge(rsi, left_index = True, right_index = True)
    return output

# Deep Learning Models
def get_train_test_data(hist_prices, split_frac = .95, days_forward = 1):
    """
    Function inputs the historical prices and returns the train-test split, together with technical indicators
    
    :param hist_prices: DataFrame containing the historical prices of a stock. Must only have 1 column
    :param split_frac: np.float64 The fraction of data that will be given to assigned as training data. 1 - split_frac = amount of test data
    :param days_forward: int - number of days in the future the model is supposed to predict, default 1 (next day's price)
    
    :returns: A tuple containing arrays, each array containing the normalized prices for training, technical indicators for training, 
            next day prices for training, prices for testing, and technical indicators for testing, next day prices for testing,
            and the price_normalizer instance required to reverse transform prices
            (prices_train, ti_train, nxt_day_prices_train, prices_test, ti_test, nxt_day_prices_test, price_noramlizer)
    """
    
    assert hist_prices.shape[1] == 1
    
    ti = calc_technical_indicators(hist_prices)
    ticker = hist_prices.columns[0]
    
    prices = hist_prices[hist_prices.index >= ti.index[0]] #using absolute prices
    prices = prices[prices.index >= ti.index[0]] #match shape of our technicals
    
    
    # normalize the price
    price_normalizer = preprocessing.MinMaxScaler() 
    prices_normalized = price_normalizer.fit_transform(prices)
    nxt_day_prices_norm = np.roll(prices_normalized, -days_forward)[:-days_forward] #days forward to predict
    
    # truncate input prices range
    prices_normalized = prices_normalized[:-days_forward]
    ti = ti[:-days_forward]

    # normalize technical indicators
    ti_normalizer = preprocessing.MinMaxScaler()
    ti_normalized = ti_normalizer.fit_transform(ti)
    
    # Train-test split
    split_time = int(split_frac * prices_normalized.shape[0]) #get amount of training data
    
    prices_train = prices_normalized[: split_time] #y
    prices_train = prices_train.reshape(prices_train.shape[0], 1, 1) 
    ti_train = ti_normalized[: split_time]
    ti_train = ti_train.reshape(ti_train.shape[0], ti_train.shape[1], 1)
    
    nxt_day_prices_train = nxt_day_prices_norm[:split_time]
    nxt_day_prices_train = nxt_day_prices_train.reshape(nxt_day_prices_train.shape[0], 1, 1) #y

    prices_test = prices_normalized[split_time: ]
    prices_test = prices_test.reshape(prices_test.shape[0], 1, 1) #y
    ti_test = ti_normalized[split_time: ]
    ti_test = ti_test.reshape(ti_test.shape[0], ti_test.shape[1], 1)
    nxt_day_prices_test = nxt_day_prices_norm[split_time: ]
    
    # Ensure data is in the correct dimensions
    # print("--- Training Data ---")
    # print("Prices Train: " + str(prices_train.shape))
    # print("Ti Train: " + str(ti_train.shape))

    # print("--- Test Data ---")
    # print("Prices Test: " + str(prices_test.shape))
    # print("Ti Test: " + str(ti_test.shape))
    
    return prices_train, ti_train, nxt_day_prices_train, prices_test, ti_test, nxt_day_prices_test, price_normalizer

# def train_LSTM_model(hist_prices, split_frac = 0.95, days_forward = 1):
#     """
#     Function inputs historical prices and a split fraction to get the train-test split from get_train_test_split, then
#     trains the data on an LSTM model. Returns the trained model
    
#     :param hist_prices: DataFrame - containing the historical prices of a stock. Must only have 1 column
#     :param split_frac: np.float64 - The fraction of data that will be given to assigned as training data. 1 - split_frac = amount of test data
#     :param days_forward: int - number of days in the future the model is supposed to predict, default 1 (next day's price)
    
#     :returns: keras.engine.functional.Functional - The trained LSTM model 
#     """
    
#     # Get necessary datasets for model training
#     prices_train, ti_train, nxt_day_prices_train, prices_test, ti_test, nxt_day_prices_test, price_normalizer = get_train_test_data(hist_prices, split_frac)
    
#     # Tensor input
#     prices_input = Input(shape = (prices_train.shape[1], prices_train.shape[2]))
#     ti_input = Input(shape = (ti_train.shape[1], ))
    
    
#     # Specify LSTM branch - use time series data
#     x = LSTM(32, return_sequences = False)(prices_input)
#     x = Dropout(.2)(x)
#     lstm_branch = Model(inputs = prices_input, outputs = x)
    
#     # Specify Dense branch - use technical data 
#     # Note: This was done after tuning, performance better than using LSTM
#     x2 = Dense(32)(ti_input)
#     x2 = Dropout(.2)(x2)
#     ti_branch = Model(inputs = ti_input, outputs = x2)
    
#     # Combine both branches
#     combined_model = concatenate([lstm_branch.output, ti_branch.output])
#     x3 = Dense(64, activation = 'sigmoid')(combined_model) # Compile with sigmoid function
#     x3 = Dense(1, activation = "linear")(x3) # Prediction

#     model = Model(inputs = [lstm_branch.input, ti_branch.input], outputs = x3)
    
#     opt = optimizers.Adam(lr = .0005)
#     model.compile(optimizer = opt, loss = "mse", metrics = ['mae'])
#     callback = callbacks.EarlyStopping(monitor = 'loss', patience = 5)
#     history_callback = model.fit(x = [prices_train, ti_train] , y = nxt_day_prices_train, batch_size = 32, epochs = 100, callbacks = callback)
#     plt.plot(history_callback.history['loss'])
    
#     return model

# def test_LSTM_model(model, hist_prices, split_frac = 0.95, return_real_prices = False):
#     """
#     Function inputs the trained model, historical prices and a split fraction to get the train-test split from get_train_test_split, then
#     trains the data on an LSTM model. Returns the predicted prices
    
#     :param model: keras.engine.functional.Functional - The trained LSTM model 
#     :param hist_prices: DataFrame - containing the historical prices of a stock. Must only have 1 column
#     :param split_frac: np.float64 - The fraction of data that will be given to assigned as training data. 1 - split_frac = amount of test data
#     :param return_real_prices: Boolean - Default False to only return the predicted prices. True to return a tuple (predicted_prices, real_prices)

#     :returns: if return_real_prices False - tuple (np.darray, pd.DataFrame, np.float64) - containing predicted prices, Array is 2D,
#                                             pd.DataFrame containing dates of the predicted values, and np.float64 the MeanAbsolutePercentageError
#               if return_real_prices True - tuple (np.darray, np.darray, pd.DataFrame, np.float64)- 4 elements, 2 are 2D np.darray (predicted_prices, real_prices), 
#                                             1 pd.DataFrame containing dates of the predicted values, and np.float64 the MeanAbsolutePercentageError
#     """
#     n = int(split_frac * hist_prices.shape[0]) 
#     test_dataset = hist_prices[n-1: ]
#     test_dataset = test_dataset.reset_index()
#     dates = test_dataset.iloc[:,[0]]

#     # Get necessary datasets for model training
#     prices_train, ti_train, nxt_day_prices_train, prices_test, ti_test, nxt_day_prices_test, price_normalizer = get_train_test_data(hist_prices, split_frac)
    
#     # Get predictions
#     y_test_predicted = model.predict([prices_test, ti_test])
#     y_test_predicted = price_normalizer.inverse_transform(y_test_predicted)
#     y_test_real = price_normalizer.inverse_transform(prices_test.reshape(prices_test.shape[0], 1))

#     # print("RMSE: " +  str(tf.keras.losses.MeanSquaredError()(y_test_real, y_test_predicted)))
#     #real = plt.plot(prices_test_unscaled, label = "Real")
#     # real = plt.plot(y_test_real, label = "Real")
#     # pred = plt.plot(y_test_predicted, label = "Predicted")
#     # plt.legend(["Real", "Predicted"])
#     # plt.show()
#     mape = losses.MeanAbsolutePercentageError()(y_test_real, y_test_predicted).numpy()
    
#     if return_real_prices:
#         return y_test_predicted, y_test_real, dates, mape
#     else:
#         return y_test_predicted, dates, mape

# def predict_prices(model, prices_test, days_forward = None):
#     """
#     Function inputs the trained model, historical prices to test on and the technical indictors 
#     Returns the predicted values of prices 
    
#     :param model: keras.engine.functional.Functional - The trained LSTM model 
#     :param prices_test: np.array - prices to test on.
#     :param days_forward: int - Default None - returns all predicted prices, else return only the set number of days  
    
#     :returns: np.darray - array containing predicted prices. Array is 2D
#     """
    
#     assert len(prices_test) >= 30, "Dataset is not long enough for model to make a prediction"
    
#     ti = calc_technical_indicators(prices_test)
    
#     prices = prices_test[prices_test.index >= ti.index[0]] #using absolute prices
#     prices = prices[prices.index >= ti.index[0]] #match shape of our technicals
    
    
#     # normalize the price
#     price_normalizer = preprocessing.MinMaxScaler() 
#     prices_normalized = price_normalizer.fit_transform(prices)

#     # normalize technical indicators
#     ti_normalizer = preprocessing.MinMaxScaler()
#     ti_normalized = ti_normalizer.fit_transform(ti)
    
#     # reshape our inputs 
#     prices_normalized = prices_normalized.reshape(prices_normalized.shape[0], 1, 1)
#     ti_normalized = ti_normalized.reshape(ti_normalized.shape[0],ti_normalized.shape[1], 1)
    
#     y_test_predicted = model.predict([prices_normalized, ti_normalized])
#     y_test_predicted = price_normalizer.inverse_transform(y_test_predicted)
#     # print(len(y_test_predicted))
#     if isinstance(days_forward,int) and days_forward is not None:
#         if len(y_test_predicted) > days_forward:
#             return y_test_predicted[:days_forward]
#     return y_test_predicted


# ## Black Litterman Model Functions
# def get_expected_returns_from_lstm(model, hist_prices, ticker, lookback_period = 90, data_frequency = 252, annualize = True, days_forward = None):
#     """
#     Function uses a pre-trained model to predict future prices, and returns the mean return rate of the security 
    
#     :param model: keras.engine.functional.Functional - The trained LSTM model 
#     :param hist_prices: pd.DataFrame - The dataframe containing historical prices
#     :param ticker: str - the ticker of the security as found in hist_prices.columns 
#     :param lookback_period: int - the number of days the model should lookback to predict prices, as in predict_prices() 
#     :param data_frequency: int - the granularity of the hist_prices. (E.g if daily, use default 252, if monthly, use 12)
#     :param annualize: Boolean - Default True to return annualized expected returns. Else return returns of granularity same as hist_prices
#     :param days_forward: int - number of days of prices to predict
    
#     :returns: np.float64 - the expected mean returns. 
#     """
    
#     assert ticker in hist_prices.columns,  "Ticker not found in hist_prices"
    
#     expected_prices = predict_prices(model, hist_prices.filter([ticker]).iloc[-lookback_period:], days_forward)
#     expected_returns = get_returns_from_prices(pd.DataFrame(expected_prices), log_prices = True)
#     if annualize:
#         mean_ret = get_annualized_returns(expected_returns, data_frequency).values[0]
#     else:
#         mean_ret = expected_returns.values.mean()
#     return mean_ret

# def get_model_relative_views(ticker_list, 
#                     hist_prices, ticker, lookback_period = 90, data_frequency = 252, annualize = True):
#     """
#     Function computes the expected relative out/under performance of a security compared to a pre-specified list of securities 
    
#     :params ticker_list: np.array list of ticker symbols in string to be compared against 
#     :param hist_prices: pd.DataFrame - The dataframe containing historical prices
#     :param ticker: str - the ticker of the security used as an "anchor" to be compared against, as found in hist_prices.columns 
#     :param lookback_period: int - the number of days the model should lookback to predict prices, as in predict_prices() 
#     :param data_frequency: int - the granularity of the hist_prices. (E.g if daily, use default 252, if monthly, use 12)
#     :param annualize: Boolean - Default True to return annualized expected returns. Else return returns of granularity same as hist_prices
    
#     :returns: pd.DataFrame - shape(n, 1) where n is the number of securities in ticker_list. Contains the relative performance. 
#     positive values imply that the anchor outperformed relative to the ticker in the row, and vice versa
#     """
#     rel_expected_ret = pd.DataFrame(columns = [ticker])
#     # set the ticker's expected returns which we will use to compare with
#     ticker_model = models.load_model("lstm_models/f" + ticker + "_lstm_model.h5")
#     ticker_expected_ret = get_expected_returns_from_lstm(ticker_model, hist_prices, ticker, lookback_period, data_frequency, annualize)
#     for i in range(len(ticker_list)):
#         tick = ticker_list[i]
#         # model = model_list[i]
#         model = models.load_model("lstm_models/f" + tick + "_lstm_model.h5")
#         rel_ret = ticker_expected_ret - get_expected_returns_from_lstm(model, hist_prices, tick, lookback_period, data_frequency, annualize) 
#         rel_expected_ret = rel_expected_ret.append({ticker: rel_ret}, ignore_index = True)
    
#     rel_expected_ret.index = ticker_list
#     return rel_expected_ret

# def get_model_views_matrix(ticker_list, hist_prices, lookback_period = 90, data_frequency = 252, annualize = True):
#     """
#     Function computes the expected relative out/under performance of each security to every other security in the ticker_list
    
#     :params ticker_list: np.array list of ticker symbols in string to be compared against 
#     :param hist_prices: pd.DataFrame - The dataframe containing historical prices
#     :param lookback_period: int - the number of days the model should lookback to predict prices, as in predict_prices() 
#     :param data_frequency: int - the granularity of the hist_prices. (E.g if daily, use default 252, if monthly, use 12)
#     :param annualize: Boolean - Default True to return annualized expected returns. Else return returns of granularity same as hist_prices
    
#     :returns: pd.DataFrame - shape(n, n) where n is the number of securities in ticker_list. Contains the relative performance. 
#     positive values imply that the anchor outperformed relative to the ticker in the row, and vice versa
#     """
#     views_matrix = pd.DataFrame
#     i = 0 
#     for ticker in ticker_list:
#         rel_views = get_model_relative_views(ticker_list, 
#                     hist_prices, ticker, lookback_period, data_frequency, annualize)
#         if i == 0:
#             i = 1
#             views_matrix = rel_views
#         else:    
#             views_matrix = views_matrix.merge(rel_views, left_index = True, right_index = True)
        
#     return views_matrix

def get_model_views_matrix_arc(ticker_list, rel_perf_data):
    """
    Function returns the views_matrix as in get_model_views_matrix(), but uses pre-saved data to filter instead of recomputing 

    :params ticker_list: np.array list of ticker symbols in string to be compared against
    :param rel_perf_data: pd.DataFrame a (n,n) matrix containing the relative outperformance as computed from get_model_views_matrix() on all avail stocks

    :returns: pd.DataFrame - shape(n, n) where n is the number of securities in ticker_list. Contains the relative performance. 
    positive values imply that the anchor outperformed relative to the ticker in the row, and vice versa
    """
    rel_perf_data = rel_perf_data.filter(ticker_list)[rel_perf_data.index.isin(ticker_list)]
    rel_perf_data = rel_perf_data[rel_perf_data.index] # make sure matrices is square

    return rel_perf_data


def get_model_views_and_link_matrices(views_matrix):
    """
    Function generates the link and views matrices as required in get_black_litterman_optimization()  
    
    :params views_matrix: pd.DataFrame - the return value from get_model_views_matrix()
    
    :returns: tuple - a tuple containing 2 arrays - 1st the views_matrix as a 1D array, and the link_matrix as a 2D array
    
    """
    views_list = []
    link_list = []
    i = 0 #row index
    for index, row in views_matrix.iterrows():
        j = 0 # column index 
        for j in range(len(row)):
            link_row = [0] * len(views_matrix.columns)
            if row[j] > 0: #if column outperforms row 
                link_row[i] = -1 
                link_row[j] = 1
                views_list.append(abs(row[j]))
                link_list.append(link_row)
            j += 1
        i += 1
    return np.array(views_list), np.array(link_list)