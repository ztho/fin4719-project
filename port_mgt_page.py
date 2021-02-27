import streamlit as st
import data_loader as data 
import user_state as u 
import workers 

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models import NumeralTickFormatter, HoverTool
import numpy as np

stock_data = data.get_stock_data()
rel_perf_data = data.get_rel_perf()

def app(tar_stocks):

    # Sidebar https://github.com/antonio-catalano/StockAnalysisApp/blob/master/app.py
    days_sim = st.sidebar.number_input("Number of Simulation Days", value = 30)
    init_cap = st.sidebar.number_input("Initial Capital", value = 1000)
    rf = st.sidebar.number_input("Risk Free Rate", value  = .0164)
    risk_level = st.sidebar.number_input("VaR Significance", value = .05)

    u.tar_stocks = st.sidebar.multiselect(label = "Selected Stocks" , options = stock_data.columns)

    
    
    opt_weights = workers.exec_get_black_litterman_optimal_weights(u.tar_stocks, stock_data, rel_perf_data)
    # st.write(opt_weights)
    sim = workers.exec_run_cppi(opt_weights, stock_data,u.tar_stocks, months_back = 6)
    sim_gbm, sim_VaR = workers.exec_simulate_gbm(sim, days_sim, init_cap, rf, risk_level = risk_level)

    # Main Display 
    st.title("Portfolio Management")
    with st.beta_container():
        col1, col2 = st.beta_columns([4, 1])
        with col1:
            if sim is not None:
                st.markdown("### Portfolio Historical Performance")
                st.bokeh_chart(workers.show_portfolio_backtest_plot(sim))
            df = workers.show_optimal_weights(opt_weights, u.tar_stocks)
            if df is not None:
                st.table(workers.show_optimal_weights(opt_weights, u.tar_stocks))
            else:
                st.write("No Allocation Feasible")

        with col2:
            st.markdown("### Portfolio Statistcs")
            
            st.markdown("#### Historical Returns")
            # st.markdown(workers.show_portfolio_returns(opt_weights, stock_data, u.tar_stocks))
            st.markdown(workers.show_portfolio_returns2(sim))
            st.markdown("#### Historical Volatility")
            st.markdown(workers.show_portfolio_vols(opt_weights, stock_data, u.tar_stocks))
            
            st.markdown("#### Value at Risk, " + str(risk_level * 100) + "%")
            st.markdown(workers.show_formatted_VaR(sim_VaR))

            st.markdown("#### % Times Profitable")
            st.markdown(workers.show_is_profitable(sim_gbm))

    with st.beta_container():
        st.markdown("#### Portfolio Projections")
        col3, col4 = st.beta_columns([4, 1])
        with col3:
            if sim_gbm is not None:
                st.bokeh_chart(workers.show_portfolio_future_plot(sim_gbm, init_cap, days_sim, stock_data))
           


